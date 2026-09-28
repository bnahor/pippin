# Pippin

**Decoupled, pipelined robot simulation: physics on CPU cores, rendering and
inference on the GPU.**

Pippin is an open-source simulator for robot learning in the spirit of
ManiSkill and Isaac Lab. It is built around one observation: **contact-rich
physics does not scale well on GPUs.**

Every environment has a different number of contacts and needs a different
number of solver iterations. GPU threads run in lockstep, so each group of
threads waits for its slowest environment, and the cost compounds. In our own
Metal physics backend, Ant runs at 20M steps/s without contacts and falls to
1.6–6.6M with them.

So Pippin splits the work:

- **Physics runs on CPU cores.** Backends plug in behind one trait: Pippin's
  own engine, which matches MuJoCo through contact, or MuJoCo itself.
- **Rendering and policy inference run on the GPU.** A batched Metal renderer
  is built in. On NVIDIA hardware, [ovrtx] is planned.
- **The two overlap.** An async, pipelined vector env splits environments into
  groups. While CPU cores step one group, the GPU renders and runs the policy
  for another.

```text
physics (CPU) : [A1][B1]    [A2][B2]    ...
render  (GPU) :     [A1][B1]    [A2][B2]
policy  (GPU) :         [A1][B1]    [A2]
```

> Status: **early (v0.1)**. APIs will change.

## Architecture

| crate | role |
|---|---|
| `pippin-env` | `Physics` and `Renderer` traits, `AsyncEnv` pipeline, Pippin CPU backend |
| `pippin` | Pippin physics engine: CPU reference, validated against MuJoCo |
| `pippin-mujoco` | MuJoCo backend via its C API (bindgen). Uses `MUJOCO_DIR` or the `mujoco` pip wheel. |
| `pippin-render` | Batched Metal rasterizer: RGB, linear depth, segmentation |
| `pippin-metal` | Pippin physics as Metal kernels, for contact-light workloads |
| `pippin-py` | Python bindings: `pippin.AsyncEnv`, `pippin.Sim` |

Physics backends only export a static scene (shapes, meshes, cameras) and
per-environment poses. Renderers consume only those, so any physics engine
works with any renderer.

## End-to-end results (M5 Pro: 6 Super + 12 Performance cores)

Ant, 2048 envs, 64×64 RGB rendered for every action, 4 physics substeps per
action, NatureCNN policy on the GPU (PyTorch MPS). Run with
`python examples/vision_policy_loop.py`:

| execution | actions/s |
|---|---|
| synchronous (1 group) | 118k |
| pipelined (2 groups) | **174k** |

Batched rendering alone (4096 Ant envs, RGB + depth + segmentation):
**310k frames/s at 64×64** and **195k frames/s at 128×128**.

[ovrtx]: https://developer.nvidia.com/omniverse

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

# pipelined vision env: CPU physics + GPU rendering
env = pippin.AsyncEnv("assets/ant.xml", 2048, backend="pippin",  # or "mujoco"
                      groups=2, substeps=4,
                      render=dict(width=64, height=64,
                                  views=[dict(eye=[1.5, -1.5, 1.2], target=[0, 0, 0.3])]))
env.reset()
while True:
    obs = env.recv()                     # next finished group
    rgb = obs["rgb"]                     # (n, views, H, W, 4) uint8, zero-copy
    actions = policy(rgb)                # e.g. on the GPU
    env.send(obs["group"], actions)      # (n, nu) float64

# plain batched physics
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

1. ~~Newton constraint solver~~ (matches MuJoCo through contact)
2. ~~Decoupled physics/rendering, async pipeline, batched Metal renderer~~
3. **ovrtx renderer backend** for photorealistic RTX rendering on NVIDIA
   hardware (Linux).
4. **CPU performance on Apple Silicon.** NEON SIMD across environments,
   thread QoS for the two core tiers, and `target-cpu` tuning.
5. **Meshes and assets.** Convex-hull collision (GJK/EPA), URDF, and
   YCB-style objects.
6. **Manipulation essentials.** Elliptic cones, torsional friction, equality
   constraints, tendons, and sensors.
7. **Zero-copy handoff to PyTorch-MPS and MLX tensors,** plus a Gymnasium
   wrapper and a ManiSkill-style task suite.

## Layout

```
crates/pippin         physics engine (Rust)
crates/pippin-env     Physics/Renderer traits, AsyncEnv pipeline
crates/pippin-mujoco  MuJoCo backend
crates/pippin-render  batched Metal renderer
crates/pippin-metal   GPU physics kernels
crates/pippin-py      Python bindings (PyO3)
examples/             end-to-end Python examples
python/pippin      Python package
assets/            test and benchmark models
validation/        MuJoCo cross-checks
benches/           performance benchmarks
```

## License

Apache-2.0
