//! GPU throughput: `cargo run --release -p pippin-metal --example gpu_bench -- assets/ant.xml`
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "assets/ant.xml".into());
    let m = pippin::mjcf::load_file(&path).expect("load");
    for n in [1024, 4096, 16384, 65536] {
        let mut sim = pippin_metal::MetalSim::new(m.clone(), n).expect("metal");
        for (i, c) in sim.ctrl_mut().iter_mut().enumerate() {
            *c = ((i % 97) as f32 * 0.37).sin();
        }
        sim.step(5).unwrap(); // warm-up
        for substeps in [1, 10] {
            let calls = 200 / substeps;
            let t = Instant::now();
            for _ in 0..calls {
                sim.step(substeps).unwrap();
            }
            let dt = t.elapsed().as_secs_f64();
            let rate = (n * calls * substeps) as f64 / dt;
            println!("{path} envs {n:>6} steps/dispatch {substeps:>2}: {:>8.1}M steps/s", rate / 1e6);
        }
    }
}
