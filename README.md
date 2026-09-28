# Pippin

**GPU-batched robot simulation for Apple Silicon.**

Pippin is an open-source physics simulator for robot learning, in the spirit of
ManiSkill / Isaac Lab, built natively for Macs. Isaac Sim does not run on macOS,
and MuJoCo's GPU paths target CUDA. Pippin's goal is to run thousands of parallel
manipulation environments on the M-series GPU through Metal, using unified
memory so observations reach PyTorch/MLX without being copied.

> Status: **early (v0.1)**. The CPU reference engine is working and validated.
> The Metal backend is next. APIs will change.

## What works today

- **MJCF loading.** Bodies, hinge/slide/ball/free joints, default classes and
  `childclass`, all orientation formats (`quat`, `euler`, `axisangle`,
  `xyaxes`, `zaxis`), `fromto`, inertia from geoms, `<inertial>`, and
  motor/position/velocity/general actuators.
- **Dynamics.** Reduced coordinates using CRBA for the mass matrix and RNEA for
  bias forces, with armature, damping, stiffness, and a semi-implicit Euler
  integrator with implicit damping.
- **Collision.** Plane, sphere, capsule, and box with every pairing, plus
  cylinder against plane. Box-box uses SAT with face clipping for stable
  multi-point manifolds.
- **Contact solver.** Soft projected Gauss-Seidel with friction, joint limits,
  warm starting, and split-impulse relaxation.
- **Batched API.** Rust `Batch` and Python `pippin.Sim`, with multithreaded
  stepping on CPU.

## Validation against MuJoCo

`validation/compare_mujoco.py` loads the same MJCF into Pippin and MuJoCo 3.x and compares:

| check | result |
|---|---|
| body masses | match to 1e-16 |
| joint-space inertia M(q), random states | match to ~1e-16 (relative) |
| bias forces c(q, q̇), random states | match to ~1e-15 (relative) |
| 2-second trajectories under random control | match to ~1e-15 |

The test models cover hinge, slide, ball, and free joints, armature, damping,
joint `ref`, and all actuator types. Contact models differ from MuJoCo's by
design, so contact is tested against analytic physics in
`crates/pippin/tests/contact.rs`:

- rest heights
- a 10-box twisted tower that stays stable for 10 s
- stick below the friction angle
- sliding acceleration matching `g(sin θ − μ cos θ)` to 13 digits
- joint limits

## Performance (CPU reference, M5 Pro, 1024 envs)

| model | MuJoCo (rollout, 18 threads) | Pippin CPU | ratio |
|---|---|---|---|
| Ant | 3.6M steps/s | 2.0M steps/s | 0.55× |
| 3D arm | 9.8M steps/s | 5.6M steps/s | 0.57× |
| Cartpole | 15.4M steps/s | 6.7M steps/s | 0.44× |

The CPU engine is a readable reference, not the fast path. Its known gaps are
PGS convergence on heavily coupled systems (fixed by the Newton solver) and
per-call thread wake-up. The performance target is the Metal backend.

## Quick start

```bash
uv venv && uv pip install maturin numpy mujoco
uv run maturin develop --release
```

```python
import numpy as np
import pippin

sim = pippin.Sim("assets/ant.xml", num_envs=4096)
for _ in range(1000):
    sim.set("ctrl", np.random.uniform(-1, 1, (sim.num_envs, sim.nu)))
    sim.step()
qpos = sim.get("qpos")  # (4096, nq)
```

Rust tests: `cargo test --release`.
Validation: `python validation/compare_mujoco.py`.
Benchmark: `python benches/bench_cpu.py`.

## Roadmap

1. **Newton constraint solver.** A convex, MuJoCo-style solver for fast,
   accurate convergence on coupled contact-rich systems.
2. **Metal backend.** Batched kernels mirroring the CPU layout, zero-copy
   buffers, and a head-to-head benchmark against MJX and Genesis on Apple
   Silicon.
3. **Meshes.** Convex-hull collision with GJK/EPA, support for URDF and
   MJCF `<asset>`, and YCB-style objects.
4. **Manipulation essentials.** Torsional friction, equality constraints
   (welds and mimic grippers), tendons, and sensors.
5. **Rendering.** A batched Metal rasterizer for RGB, depth, and segmentation
   cameras.
6. **RL layer.** Gymnasium vector env, ManiSkill-style task suite, and
   PPO baselines.

## Layout

```
crates/pippin      core engine (Rust)
crates/pippin-py   Python bindings (PyO3)
python/pippin      Python package
assets/            test and benchmark models
validation/        MuJoCo cross-checks
benches/           performance benchmarks
```

## License

Apache-2.0
