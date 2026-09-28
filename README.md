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
- **Constraint solvers.**
  - *Newton* (the default) is a convex primal Newton solver with pyramidal
    friction cones and MuJoCo's `solref`/`solimp`/`impratio` soft-constraint
    model. It uses an exact line search and usually converges in 1-3
    iterations.
  - *PGS* is a soft projected Gauss-Seidel solver with warm starting and
    split-impulse relaxation.
- **Batched API.** Rust `Batch` and Python `pippin.Sim`, with multithreaded
  stepping and open-loop `rollout()` on CPU.

## Validation against MuJoCo

`validation/compare_mujoco.py` loads the same MJCF into Pippin and MuJoCo 3.x
and compares the results. Pippin's Newton solver uses the same constraint
formulation as MuJoCo, so trajectories agree **including through contact**:

| check | result |
|---|---|
| body masses, `invweight0` constants | match to ~1e-15 |
| joint-space inertia M(q), random states | match to ~1e-16 (relative) |
| bias forces c(q, q̇), random states | match to ~1e-15 (relative) |
| contact-free trajectories (pendulum, cartpole, 3D arm, free body) | match to ~1e-15 |
| box, sphere, and capsule dropped on a plane (2 s) | match to ~5e-12 |
| Ant under random control, through ground contact (1 s) | match to ~4e-13 |

Box-box contact uses a different collision algorithm from MuJoCo's, so box
stacks agree only to about 0.5 mm. `crates/pippin/tests/contact.rs` checks
contact against analytic physics under both solvers:

- rest heights
- a stable box stack
- stick below the friction angle
- sliding acceleration matching `g(sin θ − μ cos θ)`
- joint limits

## Performance (CPU reference, M5 Pro, 1024 envs × 200 steps)

Open-loop rollouts under random control. Both simulators do the full loop
natively, using 18 threads.

| model | MuJoCo `rollout` | Pippin `rollout` | ratio |
|---|---|---|---|
| Ant (contacts, joint limits) | 3.3M steps/s | 4.8M steps/s | 1.4× |
| 3D arm | 9.8M steps/s | 19.2M steps/s | 2.0× |
| Cartpole | 16.3M steps/s | 43.7M steps/s | 2.7× |

MuJoCo's rollout also records full state and sensor data, so it does more
output work. A per-step Python loop (`set` + `step`) is slower (2.4M
steps/s on Ant) because of per-call thread wake-up. The Metal backend is the
fast path for RL.

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

# open-loop: whole control sequence in one call
traj = sim.rollout(np.zeros((sim.num_envs, 200, sim.nu)))  # (4096, 200, nq)
```

Rust tests: `cargo test --release`.
Validation: `python validation/compare_mujoco.py`.
Benchmark: `python benches/bench_cpu.py`.

## Roadmap

1. ~~**Newton constraint solver.**~~ Done. It matches MuJoCo through contact.
2. **Metal backend.** Batched kernels mirroring the CPU layout, zero-copy
   buffers, and a head-to-head benchmark against MJX and Genesis on Apple
   Silicon.
3. **Meshes.** Convex-hull collision with GJK/EPA, support for URDF and
   MJCF `<asset>`, and YCB-style objects.
4. **Manipulation essentials.** Elliptic cones, torsional friction,
   equality constraints (welds and mimic grippers), tendons, and sensors.
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
