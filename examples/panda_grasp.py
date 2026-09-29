"""Franka Panda picks up a randomized cube in every environment.

Scene from the composer (table + Menagerie Panda + one cube with random
size, mass, friction, color, and position), joint targets from a small
damped-least-squares IK, executed by the Panda's position actuators.

    scripts/fetch_assets.sh
    python examples/panda_grasp.py --envs 16
"""

import argparse

import numpy as np

import pippin
from pippin.scene import Box, SceneBuilder

PANDA = "third_party/mujoco_menagerie/franka_emika_panda/panda.xml"
HOME = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
TCP = 0.1034  # hand origin to fingertip center, along the hand z axis
TOP = 0.4


def scene(n, seed):
    sb = SceneBuilder()
    sb.floor()
    sb.table(size=(0.5, 0.45), height=TOP)
    sb.robot(PANDA, pos=(-0.45, 0.0, TOP))
    sb.objects([Box(size=((0.018, 0.024),) * 3, density=(300, 800), friction=(0.8, 1.2))],
               count=1, region=((-0.02, 0.12), (-0.12, 0.12)), surface=TOP)
    return sb.build(n, seed=seed)


def quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Kinematics:
    """Hand pose for arm joint angles, on a private single-env simulator."""

    def __init__(self, xml):
        self.sim = pippin.Sim(xml, 1)
        self.hand = self.sim.body_names.index("hand")
        self.q = self.sim.get("qpos")

    def fk(self, arm):
        q = self.q.copy()
        q[0, :7] = arm
        self.sim.set("qpos", q)
        self.sim.forward()
        pos = self.sim.body_pos()[0, self.hand]
        mat = quat_mat(self.sim.body_quat()[0, self.hand])
        return pos + TCP * mat[:, 2], mat

    def ik(self, target, rot, q0, iters=100):
        q = q0.copy()
        for _ in range(iters):
            p, r = self.fk(q)
            err = np.concatenate([target - p, 0.5 * sum(np.cross(r[:, i], rot[:, i]) for i in range(3))])
            if np.linalg.norm(err) < 1e-5:
                break
            J = np.zeros((6, 7))
            for k in range(7):
                dq = q.copy()
                dq[k] += 1e-6
                p2, r2 = self.fk(dq)
                J[:3, k] = (p2 - p) / 1e-6
                J[3:, k] = 0.5 * sum(np.cross(r[:, i], r2[:, i]) for i in range(3)) / 1e-6
            q += J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
        return q


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-open", action="store_true", help="negative control: never close the gripper")
    p.add_argument("--render", default="", help="save a contact sheet of the lifted cubes (PPM)")
    args = p.parse_args()
    n = args.envs
    plan = scene(n, args.seed)
    info = pippin._pippin.model_info(plan.xml)
    cube_q = info["joint_qposadr"][info["joint_names"].index("obj0_joint")]

    # start every env at the Panda's home pose with the gripper open
    plan.qpos0[:, :7] = HOME
    plan.qpos0[:, 7:9] = 0.04
    render = dict(width=200, height=150, views=[dict(eye=[0.55, -0.6, 0.85], target=[0.0, 0.0, 0.5], fovy=50)])
    env = pippin.AsyncEnv(plan.xml, n, groups=1, substeps=10, render=render if args.render else None)
    plan.apply(env)
    obs = env.recv()

    kin = Kinematics(plan.xml)
    _, down = kin.fk(HOME)  # keep the home orientation (gripper pointing down)
    cubes = obs["qpos"][:, cube_q:cube_q + 3]
    waypoints = []  # per env: arm targets for pre-grasp, grasp, lift
    for e in range(n):
        c = cubes[e]
        pre = kin.ik(c + [0, 0, 0.12], down, HOME)
        grasp = kin.ik(c + [0, 0, 0.0], down, pre)
        lift = kin.ik(c + [0, 0, 0.20], down, grasp)
        waypoints.append((pre, grasp, lift))

    def run(stage, steps, gripper):
        for t in range(steps):
            a = min(1.0, (t + 1) / (0.7 * steps))
            ctrl = np.zeros((n, env.nu))
            for e in range(n):
                q0, q1 = stage(e)
                ctrl[e, :7] = (1 - a) * q0 + a * q1
            ctrl[:, 7] = gripper
            env.send(0, ctrl)
            out = env.recv()
        return out

    run(lambda e: (HOME, waypoints[e][0]), 60, 255)  # above the cube, open
    run(lambda e: (waypoints[e][0], waypoints[e][1]), 50, 255)  # descend
    close = 255 if args.keep_open else 0
    run(lambda e: (waypoints[e][1], waypoints[e][1]), 40, close)  # close
    obs = run(lambda e: (waypoints[e][1], waypoints[e][2]), 60, close)  # lift
    lifted = obs["qpos"][:, cube_q + 2] - cubes[:, 2]
    ok = lifted > 0.1
    print("cube lift per env (m):", np.round(lifted, 3).tolist())
    print(f"success: {ok.sum()}/{n} envs lifted the cube more than 10 cm")
    if args.render:
        rgb = obs["rgb"][:, 0, :, :, :3]
        cols = 4
        h, w = rgb.shape[1:3]
        sheet = np.zeros((((n + cols - 1) // cols) * h, cols * w, 3), np.uint8)
        for i in range(n):
            r, c = divmod(i, cols)
            sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = rgb[i]
        with open(args.render, "wb") as f:
            f.write(f"P6\n{sheet.shape[1]} {sheet.shape[0]}\n255\n".encode() + sheet.tobytes())


if __name__ == "__main__":
    main()
