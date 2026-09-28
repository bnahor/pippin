"""Load the same URDF in Pippin and MuJoCo (which imports URDF natively) and
compare dynamics, matching joints by name (the two may order them
differently, and MuJoCo merges fixed links).
"""

import pathlib
import sys

import mujoco
import numpy as np

import pippin

URDF = pathlib.Path(__file__).resolve().parent.parent / "assets" / "urdf" / "arm.urdf"


def main():
    mm = mujoco.MjModel.from_xml_path(str(URDF))
    md = mujoco.MjData(mm)
    sim = pippin.Sim(str(URDF), 1)
    assert (sim.nq, sim.nv) == (mm.nq, mm.nv), f"dims {(sim.nq, sim.nv)} vs {(mm.nq, mm.nv)}"

    # permutation from MuJoCo dof/qpos order to Pippin order, by joint name
    qmap, vmap = np.zeros(mm.nq, int), np.zeros(mm.nv, int)
    for j, name in enumerate(sim.joint_names):
        jm = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert jm >= 0, f"joint {name} missing in MuJoCo"
        qmap[sim.joint_qposadr[j]] = mm.jnt_qposadr[jm]
        vmap[sim.joint_dofadr[j]] = mm.jnt_dofadr[jm]
    print("joints:", sim.joint_names)
    print(f"total mass: pippin {sum(sim.body_mass()):.9f}  mujoco {mm.body_mass.sum():.9f}")

    rng = np.random.default_rng(1)
    ok = True
    for trial in range(10):
        q = mm.qpos0.copy() + rng.uniform(-0.8, 0.8, mm.nq) * 0.3
        v = rng.normal(size=mm.nv)
        md.qpos[:], md.qvel[:] = q, v
        mujoco.mj_forward(mm, md)
        full = np.zeros((mm.nv, mm.nv))
        mujoco.mj_fullM(mm, md, full)
        sim.set("qpos", q[qmap][None])
        sim.set("qvel", v[vmap][None])
        sim.forward()
        em = np.max(np.abs(sim.mass_matrix() - full[np.ix_(vmap, vmap)]))
        eb = np.max(np.abs(sim.qfrc_bias() - md.qfrc_bias[vmap]))
        ok &= em < 1e-10 and eb < 1e-10
        if trial == 0:
            print(f"M(q) err {em:.2e}   bias err {eb:.2e}")

    # trajectory from a moving start (joint limits active)
    mujoco.mj_resetData(mm, md)
    sim.reset()
    v0 = rng.normal(size=mm.nv)
    md.qvel[:] = v0
    sim.set("qvel", v0[vmap][None])
    err = 0.0
    for _ in range(1000):
        mujoco.mj_step(mm, md)
        sim.step()
        err = max(err, np.max(np.abs(sim.get("qpos")[0] - md.qpos[qmap])))
    print(f"trajectory (1000 steps, gravity + limits) max qpos err {err:.2e}")
    ok &= err < 1e-8
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
