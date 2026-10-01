"""Manipulation throughput: Panda + table + randomized cube (composer scene).

Physics only (Pippin vs MuJoCo, same CPU) and physics + 128x128 rendering
through the async pipeline.
"""

import os
import tempfile
import time

import mujoco
import numpy as np
from mujoco import rollout

import pippin
from pippin.scene import Box, SceneBuilder

N, T = 1024, 100
HOME = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255])


def scene():
    sb = SceneBuilder()
    sb.floor()
    sb.table(size=(0.5, 0.45), height=0.4)
    sb.robot("third_party/mujoco_menagerie/franka_emika_panda/panda.xml", pos=(-0.45, 0.0, 0.4))
    sb.objects([Box(size=((0.018, 0.024),) * 3)], count=1, region=((-0.02, 0.12), (-0.12, 0.12)), surface=0.4)
    plan = sb.build(N, seed=0)
    plan.qpos0[:, :7] = HOME[:7]
    plan.qpos0[:, 7:9] = 0.04
    return plan


def main():
    plan = scene()
    rng = np.random.default_rng(0)
    ctrl = np.tile(HOME, (N, T, 1)) + np.concatenate([rng.normal(0, 0.2, (N, T, 7)), np.zeros((N, T, 1))], axis=2)

    # MuJoCo: same MJCF, multithreaded rollout
    path = os.path.join(tempfile.mkdtemp(), "scene.xml")
    open(path, "w").write(plan.xml)
    mm = mujoco.MjModel.from_xml_path(path)
    md = mujoco.MjData(mm)
    md.qpos[:] = plan.qpos0[0]
    x0 = np.zeros(mujoco.mj_stateSize(mm, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(mm, md, x0, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    datas = [mujoco.MjData(mm) for _ in range(os.cpu_count())]
    with rollout.Rollout(nthread=os.cpu_count()) as r:
        r.rollout(mm, datas, np.tile(x0, (N, 1)), ctrl[:, :5])
        t = time.perf_counter()
        r.rollout(mm, datas, np.tile(x0, (N, 1)), ctrl)
        mj = N * T / (time.perf_counter() - t)

    sim = pippin.Sim(plan.xml, N)
    sim.set("qpos", plan.qpos0)
    sim.rollout(np.ascontiguousarray(ctrl[:, :5]))
    sim.set("qpos", plan.qpos0)
    sim.set("qvel", np.zeros((N, sim.nv)))
    t = time.perf_counter()
    sim.rollout(ctrl)
    pp = N * T / (time.perf_counter() - t)
    print(f"physics only, {N} envs: MuJoCo {mj/1e3:.0f}k steps/s   Pippin {pp/1e3:.0f}k steps/s   ({pp/mj:.2f}x)")

    for groups in (1, 2):
        env = pippin.AsyncEnv(plan.xml, N, groups=groups, substeps=10,
                              render=dict(width=128, height=128,
                                          views=[dict(eye=[0.9, -0.9, 1.0], target=[-0.1, 0, 0.55])]))
        plan.apply(env)
        for g in range(groups):
            env.recv()
        a = np.tile(HOME, (N // groups, 1))
        for g in range(groups):  # keep every group in flight
            env.send(g, a)
        for _ in range(2 * groups):  # warm-up
            env.send(env.recv()["group"], a)
        steps, t0 = 0, time.perf_counter()
        for _ in range(10 * groups):
            obs = env.recv()
            env.send(obs["group"], a)
            steps += N // groups
        dt = time.perf_counter() - t0
        print(f"physics (10 substeps/action) + 128x128 render, groups={groups}: "
              f"{steps / dt / 1e3:.1f}k actions/s = {10 * steps / dt / 1e3:.0f}k physics steps/s")


if __name__ == "__main__":
    main()
