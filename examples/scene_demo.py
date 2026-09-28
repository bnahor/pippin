"""Procedurally generated tabletop scenes: every environment gets its own
objects, sizes, masses, frictions, colors, and layout (seeded).

    python examples/scene_demo.py --envs 12 --out scenes.ppm
"""

import argparse

import numpy as np

import pippin
from pippin.scene import Box, Cylinder, Mesh, SceneBuilder, Sphere


def build(num_envs: int, seed: int):
    sb = SceneBuilder()
    sb.floor()
    top = sb.table(size=(0.45, 0.35), height=0.4)
    sb.robot("assets/urdf/arm.urdf", pos=(-0.38, 0.0, top))
    pool = [
        Box(size=((0.015, 0.035), (0.015, 0.035), (0.015, 0.04))),
        Cylinder(radius=(0.015, 0.03), half_height=(0.02, 0.05)),
        Sphere(radius=(0.015, 0.035)),
        Mesh("assets/urdf/meshes/wedge.obj", scale=0.8),
    ]
    sb.objects(pool, count=(2, 6), region=((-0.15, 0.4), (-0.3, 0.3)), surface=top)
    return sb.build(num_envs, seed=seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="scenes.ppm")
    args = p.parse_args()

    plan = build(args.envs, args.seed)
    env = pippin.AsyncEnv(plan.xml, args.envs, backend="pippin", groups=1, substeps=50,
                          render=dict(width=160, height=120,
                                      views=[dict(eye=[0.75, -0.75, 1.05], target=[0.05, 0.0, 0.4], fovy=45)]))
    plan.apply(env)
    obs = env.recv()  # state right after reset
    zeros = np.zeros((args.envs, env.nu))
    for _ in range(10):  # let objects settle: 10 x 50 substeps = 1 s
        env.send(0, zeros)
        obs = env.recv()
    print("objects per env:", (plan.choices >= 0).sum(axis=1).tolist())

    rgb = obs["rgb"][:, 0, :, :, :3]
    cols = 4
    rows = (args.envs + cols - 1) // cols
    h, w = rgb.shape[1:3]
    sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i in range(args.envs):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = rgb[i]
    with open(args.out, "wb") as f:
        f.write(f"P6\n{sheet.shape[1]} {sheet.shape[0]}\n255\n".encode())
        f.write(sheet.tobytes())
    print("wrote", args.out)


if __name__ == "__main__":
    main()
