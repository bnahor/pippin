"""End-to-end vision RL loop: CPU physics -> GPU render -> GPU (MPS) CNN policy.

Compares synchronous execution (1 group) with the pipelined AsyncEnv (2 groups),
where CPU physics overlaps GPU rendering and inference.

    python examples/vision_policy_loop.py --backend pippin --envs 2048
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

import pippin


class NatureCNN(nn.Module):
    def __init__(self, nu: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(256), nn.ReLU(),
            nn.Linear(256, nu), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


def run(args, groups: int) -> dict:
    env = pippin.AsyncEnv(
        args.model, args.envs, backend=args.backend, groups=groups, substeps=args.substeps,
        render=dict(width=args.res, height=args.res, views=[dict(eye=[1.5, -1.5, 1.2], target=[0, 0, 0.3])]),
    )
    dev = torch.device("mps")
    policy = NatureCNN(env.nu).to(dev).eval()
    env.reset()
    steps, t0 = 0, None
    phys = rend = infer = 0.0
    for i in range(args.iters * groups):
        obs = env.recv()
        if i == 2 * groups:  # warm-up rounds done
            t0, steps, phys, rend, infer = time.perf_counter(), 0, 0.0, 0.0, 0.0
        ti = time.perf_counter()
        x = torch.from_numpy(obs["rgb"][:, 0, :, :, :3]).to(dev).permute(0, 3, 1, 2).float().div_(255.0)
        with torch.no_grad():
            actions = policy(x).cpu().numpy().astype(np.float64)
        infer += time.perf_counter() - ti
        env.send(obs["group"], np.ascontiguousarray(actions))
        steps += actions.shape[0]
        phys += obs["physics_ms"]
        rend += obs["render_ms"]
    dt = time.perf_counter() - t0
    k = (args.iters - 2) * groups
    return dict(rate=steps / dt, phys=phys / k, rend=rend / k, infer=infer / k * 1e3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="assets/ant.xml")
    p.add_argument("--backend", default="pippin")
    p.add_argument("--envs", type=int, default=2048)
    p.add_argument("--res", type=int, default=64)
    p.add_argument("--substeps", type=int, default=4)
    p.add_argument("--iters", type=int, default=30)
    args = p.parse_args()
    print(f"{args.model} backend={args.backend} envs={args.envs} {args.res}x{args.res} rgb, "
          f"{args.substeps} physics substeps/action, NatureCNN on MPS")
    for groups in (1, 2, 4):
        r = run(args, groups)
        print(f"  groups {groups}: {r['rate']:>8.0f} actions/s   per group: physics {r['phys']:5.1f} ms, "
              f"render {r['rend']:5.1f} ms, inference {r['infer']:5.1f} ms")


if __name__ == "__main__":
    main()
