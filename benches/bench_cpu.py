"""Throughput: Pippin CPU batch vs MuJoCo multithreaded rollout, random controls."""

import os
import pathlib
import time

import mujoco
from mujoco import rollout
import numpy as np

import pippin

ASSETS = pathlib.Path(__file__).resolve().parent.parent / "assets"
NENV, NSTEP = 1024, 200
NTHREAD = os.cpu_count()


def bench(name):
    path = str(ASSETS / name)
    mm = mujoco.MjModel.from_xml_path(path)
    rng = np.random.default_rng(0)
    ctrl = rng.uniform(-1, 1, size=(NENV, NSTEP, mm.nu))

    # MuJoCo: rollout with a thread pool, one MjData per thread
    datas = [mujoco.MjData(mm) for _ in range(NTHREAD)]
    md = mujoco.MjData(mm)
    x0 = np.zeros(mujoco.mj_stateSize(mm, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(mm, md, x0, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    x0 = np.tile(x0, (NENV, 1))
    with rollout.Rollout(nthread=NTHREAD) as r:
        r.rollout(mm, datas, x0, ctrl[:, :10])  # warm-up
        t = time.perf_counter()
        r.rollout(mm, datas, x0, ctrl)
        mj = NENV * NSTEP / (time.perf_counter() - t)

    sim = pippin.Sim(path, NENV)
    sim.step(10)
    sim.reset()
    t = time.perf_counter()
    for k in range(NSTEP):
        sim.set("ctrl", np.ascontiguousarray(ctrl[:, k]))
        sim.step()
    pp = NENV * NSTEP / (time.perf_counter() - t)
    print(f"{name:<14} mujoco {mj/1e3:8.0f}k steps/s   pippin(cpu) {pp/1e3:8.0f}k steps/s   ratio {pp/mj:.2f}x")


if __name__ == "__main__":
    print(f"{NENV} envs x {NSTEP} steps, {NTHREAD} threads")
    for n in ["ant.xml", "arm3d.xml", "cartpole.xml"]:
        bench(n)
