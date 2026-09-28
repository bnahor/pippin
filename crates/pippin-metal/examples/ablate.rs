//! Time the kernel cut off after each stage (cumulative).
use std::time::Instant;
fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "assets/ant.xml".into());
    let m = pippin::mjcf::load_file(&path).unwrap();
    let lim = pippin_metal::Limits::for_model(&m);
    let names = ["full", "kinematics", "+crba", "+rne/forces", "+chol/solve", "+collision", "+rows", "+newton"];
    let n = 16384;
    for k in [1, 2, 3, 4, 5, 6, 7, 0] {
        let src = format!("#define ABLATE {k}\n{}", pippin_metal::codegen::source(&m, lim));
        let mut sim = pippin_metal::MetalSim::with_source(m.clone(), n, lim, &src).unwrap();
        for (i, c) in sim.ctrl_mut().iter_mut().enumerate() { *c = ((i % 97) as f32 * 0.37).sin(); }
        sim.step(5).unwrap();
        let t = Instant::now();
        for _ in 0..20 { sim.step(10).unwrap(); }
        let ns = t.elapsed().as_secs_f64() / (n * 200) as f64 * 1e9;
        println!("{:<12} {:>7.1} ns/env-step", names[k], ns);
    }
}
