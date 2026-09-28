use std::time::Instant;
fn main() {
    let m = pippin::mjcf::load_file("assets/ant.xml").unwrap();
    let base = pippin_metal::Limits::for_model(&m);
    println!("default limits {:?}", base);
    for (mc, mr) in [(base.max_contacts, base.max_rows), (16, 48), (8, 32), (4, 24)] {
        let lim = pippin_metal::Limits { max_contacts: mc, max_rows: mr };
        let n = 16384;
        let mut sim = pippin_metal::MetalSim::with_limits(m.clone(), n, lim).unwrap();
        sim.step(5).unwrap();
        let t = Instant::now();
        for _ in 0..20 { sim.step(10).unwrap(); }
        println!("maxcon {mc:>3} maxrows {mr:>3}: {:.1}M steps/s", (n * 200) as f64 / t.elapsed().as_secs_f64() / 1e6);
    }
}
