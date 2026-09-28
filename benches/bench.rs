//! Single-threaded step cost: `cargo run --release --example bench -- assets/ant.xml`
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "assets/ant.xml".into());
    let steps: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let m = pippin::mjcf::load_file(&path).expect("load");
    let mut d = pippin::Data::new(&m);
    let mut ncon = 0;
    let t = Instant::now();
    for i in 0..steps {
        for (u, c) in d.ctrl.iter_mut().enumerate() {
            *c = ((i as f64) * 0.01 + u as f64).sin();
        }
        pippin::step(&m, &mut d);
        ncon += d.contacts.len();
    }
    let dt = t.elapsed().as_secs_f64();
    println!("{path}: {:.2} us/step, avg contacts {:.1}", dt / steps as f64 * 1e6, ncon as f64 / steps as f64);
}
