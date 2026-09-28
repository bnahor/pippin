"""Compare Pippin against MuJoCo, with and without contact.

Checks, per model:
  1. body masses
  2. joint-space inertia M(q) and bias forces c(q, qdot) at random states
  3. full trajectories under random controls (same Euler integrator)
"""

import pathlib
import sys

import mujoco
import numpy as np

import pippin

ASSETS = pathlib.Path(__file__).resolve().parent.parent / "assets"
# (model, trajectory seconds, qpos tolerance). Contact scenes use the Newton solver.
MODELS = [
    ("pendulum.xml", 2.0, 1e-8),
    ("cartpole.xml", 2.0, 1e-8),
    ("arm3d.xml", 2.0, 1e-8),
    ("tumble.xml", 2.0, 1e-8),
    ("mesh_tumble.xml", 2.0, 1e-8),
    ("box_drop.xml", 2.0, 1e-8),
    ("ant.xml", 1.0, 1e-8),
]


def random_state(mm: mujoco.MjModel, rng):
    qpos = mm.qpos0.copy()
    qpos += 0.0
    for j in range(mm.njnt):
        adr = mm.jnt_qposadr[j]
        t = mm.jnt_type[j]
        if t == mujoco.mjtJoint.mjJNT_FREE:
            qpos[adr : adr + 3] += rng.normal(size=3) * 0.3
            q = rng.normal(size=4)
            qpos[adr + 3 : adr + 7] = q / np.linalg.norm(q)
        elif t == mujoco.mjtJoint.mjJNT_BALL:
            q = rng.normal(size=4)
            qpos[adr : adr + 4] = q / np.linalg.norm(q)
        else:
            qpos[adr] += rng.uniform(-1, 1)
    qvel = rng.normal(size=mm.nv)
    return qpos, qvel


def check(name: str, seconds: float, tol: float) -> bool:
    path = str(ASSETS / name)
    mm = mujoco.MjModel.from_xml_path(path)
    md = mujoco.MjData(mm)
    sim = pippin.Sim(path, 1)
    rng = np.random.default_rng(0)
    ok = True

    def report(label, err, tol):
        nonlocal ok
        good = err < tol
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {label:<34} max err {err:.3e}  (tol {tol:.0e})")

    assert (sim.nq, sim.nv, sim.nu) == (mm.nq, mm.nv, mm.nu), "dimension mismatch"
    report("body mass", np.max(np.abs(np.array(sim.body_mass()) - mm.body_mass)), 1e-9)
    rel = lambda a, b: np.max(np.abs(a - b) / np.maximum(1e-12, np.abs(b)))
    report("body invweight0 (relative)", rel(np.array(sim.body_invweight())[1:], mm.body_invweight0[1:]), 1e-6)
    report("dof invweight0 (relative)", rel(np.array(sim.dof_invweight()), mm.dof_invweight0), 1e-6)

    em = eb = 0.0
    for _ in range(20):
        qpos, qvel = random_state(mm, rng)
        md.qpos[:], md.qvel[:] = qpos, qvel
        mujoco.mj_forward(mm, md)
        full = np.zeros((mm.nv, mm.nv))
        mujoco.mj_fullM(mm, md, full)
        sim.set("qpos", qpos[None])
        sim.set("qvel", qvel[None])
        sim.forward()
        em = max(em, np.max(np.abs(sim.mass_matrix() - full)) / max(1.0, np.max(np.abs(full))))
        eb = max(eb, np.max(np.abs(sim.qfrc_bias() - md.qfrc_bias)) / max(1.0, np.max(np.abs(md.qfrc_bias))))
    report("mass matrix M(q) (relative)", em, 1e-10)
    report("bias forces c(q,v) (relative)", eb, 1e-10)

    # trajectory
    mujoco.mj_resetData(mm, md)
    sim.reset()
    # start moving so every inertia term matters (free fall alone does not)
    qvel0 = rng.normal(size=mm.nv) * 0.5
    md.qvel[:] = qvel0
    sim.set("qvel", qvel0[None])
    steps = int(seconds / mm.opt.timestep)
    ctrls = rng.uniform(-1, 1, size=(steps, mm.nu))
    err = 0.0
    for t in range(steps):
        if mm.nu:
            md.ctrl[:] = ctrls[t]
            sim.set("ctrl", ctrls[t][None])
        mujoco.mj_step(mm, md)
        sim.step()
        err = max(err, np.max(np.abs(sim.get("qpos")[0] - md.qpos)))
    report(f"qpos trajectory ({steps} steps)", err, tol)
    return ok


def main():
    all_ok = True
    for name, seconds, tol in MODELS:
        print(name)
        all_ok &= check(name, seconds, tol)
    print("ALL PASS" if all_ok else "SOME CHECKS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
