//! Sync vs async (pipelined) throughput with rendering and a stand-in policy.
//! `cargo run --release -p pippin-render --example pipeline_bench -- assets/ant.xml mujoco 4096 5`
//! args: model, backend (pippin|mujoco), envs, simulated policy ms per full batch
use std::time::{Duration, Instant};

use pippin_env::{AsyncEnv, Physics, PippinCpu, RenderConfig, ViewSource};
use pippin_render::MetalRenderer;

fn physics(path: &str, backend: &str, n: usize) -> Box<dyn Physics> {
    match backend {
        "mujoco" => Box::new(pippin_mujoco::MujocoPhysics::from_file(path, n).unwrap()),
        _ => Box::new(PippinCpu::from_file(path, n).unwrap()),
    }
}

fn run(path: &str, backend: &str, n: usize, groups: usize, policy_ms: f64, iters: usize) -> (f64, f64, f64) {
    let phys = physics(path, backend, n);
    let cfg = RenderConfig {
        width: 64,
        height: 64,
        views: vec![ViewSource::look_at([1.5, -1.5, 1.2], [0.0, 0.0, 0.3], 45.0)],
        near: 0.01,
        far: 20.0,
    };
    let r = MetalRenderer::new(&phys.scene(), cfg).unwrap();
    let env = AsyncEnv::new(phys, Some(Box::new(r)), groups, 4);
    let nu = env.dims().nu;
    for g in 0..groups {
        env.reset(g);
    }
    let (mut steps, mut phys_ms, mut rend_ms) = (0usize, 0.0, 0.0);
    let mut t0 = None;
    for i in 0..iters * groups {
        let obs = env.recv().unwrap();
        if i == groups {
            t0 = Some(Instant::now()); // skip warm-up round
            steps = 0;
        }
        steps += obs.envs.len();
        phys_ms += obs.timing.physics_ms;
        rend_ms += obs.timing.render_ms;
        // stand-in for GPU policy inference, proportional to group size
        std::thread::sleep(Duration::from_secs_f64(policy_ms * 1e-3 * obs.envs.len() as f64 / n as f64));
        let t = i as f64 * 0.01;
        let ctrl = (0..obs.envs.len() * nu).map(|k| ((k as f64) * 0.37 + t).sin()).collect();
        env.send(obs.group, ctrl);
    }
    let dt = t0.unwrap().elapsed().as_secs_f64();
    let k = (iters * groups) as f64;
    (steps as f64 / dt, phys_ms / k, rend_ms / k)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let path = a.get(1).cloned().unwrap_or("assets/ant.xml".into());
    let backend = a.get(2).cloned().unwrap_or("mujoco".into());
    let n: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(4096);
    let policy_ms: f64 = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(5.0);
    println!("{path} backend={backend} envs={n} 64x64 rgb+depth+seg, 4 substeps/action, policy {policy_ms} ms/batch");
    for groups in [1, 2, 4] {
        let (rate, p, r) = run(&path, &backend, n, groups, policy_ms, 20);
        println!("  groups {groups}: {:>8.0} env-steps/s (actions/s)   physics {p:6.2} ms/group   render {r:6.2} ms/group", rate);
    }
}
