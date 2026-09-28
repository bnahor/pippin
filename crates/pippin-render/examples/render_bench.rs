//! Batched render throughput: `cargo run --release -p pippin-render --example render_bench -- assets/ant.xml`
use std::time::Instant;

use pippin_env::{Physics, Pose, RenderConfig, Renderer, ViewSource};

fn main() {
    let path = std::env::args().nth(1).unwrap_or("assets/ant.xml".into());
    for (n, res) in [(1024, 64), (4096, 64), (1024, 128), (4096, 128), (1024, 256)] {
        let mut phys = pippin_mujoco::MujocoPhysics::from_file(&path, n).unwrap();
        phys.step(0..n, 50);
        let d = phys.dims();
        let mut g = vec![Pose::default(); n * d.ngeom];
        let mut c = vec![Pose::default(); n * d.ncam];
        phys.poses(0..n, &mut g, &mut c);
        let cfg = RenderConfig {
            width: res,
            height: res,
            views: vec![ViewSource::look_at([1.5, -1.5, 1.2], [0.0, 0.0, 0.3], 45.0)],
            near: 0.01,
            far: 20.0,
        };
        let mut r = pippin_render::MetalRenderer::new(&phys.scene(), cfg).unwrap();
        r.render(0, n, &g, &c, None).unwrap();
        let iters = 20;
        let t = Instant::now();
        for _ in 0..iters {
            r.render(0, n, &g, &c, None).unwrap();
        }
        let dt = t.elapsed().as_secs_f64() / iters as f64;
        println!("{n:>5} envs {res:>3}x{res:<3} rgb+depth+seg: {:>7.0} frames/s  ({:.2} ms/batch)", n as f64 / dt, dt * 1e3);
    }
}
