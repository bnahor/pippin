//! Step a physics backend, render all envs, save a contact sheet (PPM).
//! `cargo run --release -p pippin-render --example render_demo -- assets/box_drop.xml out.ppm`
use pippin_env::{Field, Physics, Pose, RenderConfig, Renderer, ViewSource};
use pippin_render::MetalRenderer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or("assets/box_drop.xml".into());
    let out = args.get(2).cloned().unwrap_or("render.ppm".into());
    let n = 4;
    let mut phys = pippin_mujoco::MujocoPhysics::from_file(&path, n).unwrap();
    let d = phys.dims();
    // perturb each env differently so the images differ
    let mut q = vec![0.0; n * d.nq];
    phys.get(Field::Qpos, 0..n, &mut q);
    for e in 0..n {
        q[e * d.nq] += 0.1 * e as f64;
    }
    phys.set(Field::Qpos, 0..n, &q);
    phys.step(0..n, 150);

    let cfg = RenderConfig {
        width: 256,
        height: 192,
        views: vec![
            ViewSource::look_at([1.2, -1.2, 0.8], [0.0, 0.0, 0.1], 45.0),
            ViewSource::look_at([0.0, 0.0, 2.0], [0.0, 0.01, 0.0], 45.0),
        ],
        near: 0.01,
        far: 20.0,
    };
    let mut r = MetalRenderer::new(&phys.scene(), cfg.clone()).unwrap();
    let mut g = vec![Pose::default(); n * d.ngeom];
    let mut c = vec![Pose::default(); n * d.ncam];
    phys.poses(0..n, &mut g, &mut c);
    let frames = r.render(n, &g, &c).unwrap();

    // contact sheet: rows = envs, cols = views
    let (w, h, nv) = (cfg.width, cfg.height, cfg.views.len());
    let (sw, sh) = (w * nv, h * n);
    let mut img = vec![0u8; sw * sh * 3];
    for e in 0..n {
        for v in 0..nv {
            for y in 0..h {
                for x in 0..w {
                    let src = (((e * nv + v) * h + y) * w + x) * 4;
                    let dst = ((e * h + y) * sw + v * w + x) * 3;
                    img[dst..dst + 3].copy_from_slice(&frames.rgba[src..src + 3]);
                }
            }
        }
    }
    let mut data = format!("P6\n{sw} {sh}\n255\n").into_bytes();
    data.extend_from_slice(&img);
    std::fs::write(&out, data).unwrap();
    let bg = frames.segmentation.iter().filter(|&&s| s < 0).count() as f64 / frames.segmentation.len() as f64;
    println!("wrote {out} ({sw}x{sh}); background fraction {bg:.2}");
}
