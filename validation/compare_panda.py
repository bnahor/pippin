"""Franka Panda (MuJoCo Menagerie) in Pippin vs MuJoCo.

Exercises <include>, default classes, materials, meshes, fixed-tendon
actuation, joint equality constraints with custom solref/solimp, and the
implicitfast integrator. Fetch the model first: scripts/fetch_assets.sh
"""

import pathlib
import sys

import mujoco
import numpy as np

import pippin

SCENE = pathlib.Path(__file__).resolve().parent.parent / "third_party/mujoco_menagerie/franka_emika_panda/scene.xml"


def main():
    if not SCENE.exists():
        print("Panda model not found; run scripts/fetch_assets.sh")
        sys.exit(0)
    mm = mujoco.MjModel.from_xml_path(str(SCENE))
    md = mujoco.MjData(mm)
    sim = pippin.Sim(str(SCENE), 1)
    assert (sim.nq, sim.nv, sim.nu) == (mm.nq, mm.nv, mm.nu)
    names = [mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(mm.njnt)]
    assert sim.joint_names == names, (sim.joint_names, names)
    ok = True

    rng = np.random.default_rng(0)
    em = eb = 0.0
    for _ in range(20):
        lo, hi = mm.jnt_range[:, 0], mm.jnt_range[:, 1]
        q = rng.uniform(lo, hi)
        v = rng.normal(size=mm.nv)
        md.qpos[:], md.qvel[:] = q, v
        mujoco.mj_forward(mm, md)
        full = np.zeros((mm.nv, mm.nv))
        mujoco.mj_fullM(mm, md, full)
        sim.set("qpos", q[None])
        sim.set("qvel", v[None])
        sim.forward()
        em = max(em, np.max(np.abs(sim.mass_matrix() - full)))
        eb = max(eb, np.max(np.abs(sim.qfrc_bias() - md.qfrc_bias)))
    print(f"M(q) err {em:.2e}   bias err {eb:.2e}")
    ok &= em < 1e-10 and eb < 1e-9

    # home keyframe, position control toward a moving target, gripper closing
    key = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(mm, md, key)
    sim.reset()  # clear the random velocities left by the checks above
    sim.set("qpos", md.qpos[None].copy())
    err, ncon = 0.0, 0
    for t in range(1500):
        ctrl = mm.key_ctrl[key].copy()
        ctrl[:7] += 0.3 * np.sin(0.004 * t + np.arange(7))
        ctrl[7] = 255 * (1 - t / 1500)
        md.ctrl[:] = ctrl
        sim.set("ctrl", ctrl[None])
        mujoco.mj_step(mm, md)
        sim.step()
        ncon = max(ncon, md.ncon)
        err = max(err, np.max(np.abs(sim.get("qpos")[0] - md.qpos)))
    print(f"3 s tracking trajectory: max qpos err {err:.2e}  (max MuJoCo contacts {ncon})")
    fingers = sim.get("qpos")[0][7:9]
    print(f"finger joints at end: {fingers}  (equality keeps them equal: diff {abs(fingers[0] - fingers[1]):.2e})")
    ok &= err < 1e-7
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
