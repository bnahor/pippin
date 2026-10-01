//! Support-query cost: brute-force scan vs graph walk, on real hulls.
//! `cargo run --release --example hull_support -- model.xml`
use parry3d_f64::math::Vector;
use parry3d_f64::shape::{ConvexPolyhedron, SupportMap};
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).expect("model");
    let m = pippin::mjcf::load_file(&path).unwrap();
    let mut sizes: Vec<_> = m.mesh_hull.iter().enumerate().map(|(i, h)| (h.vertices.len(), i)).collect();
    sizes.sort();
    sizes.dedup_by_key(|s| s.0);
    for (_, i) in sizes {
        let pts: Vec<Vector> = m.mesh_hull[i].vertices.iter().map(|v| Vector::new(v[0], v[1], v[2])).collect();
        let Some(hull) = ConvexPolyhedron::from_convex_hull(&pts) else { continue };
        let fast = pippin::fasthull::FastHull::new(hull.clone());
        let nv = hull.points().len();
        let dirs_rand: Vec<Vector> = (0..20000).map(|k| { let a = k as f64 * 0.731; Vector::new(a.sin() * (1.3 * a).cos(), (0.7 * a).cos(), (2.1 * a).sin()).normalize() }).collect();
        let dirs_coh: Vec<Vector> = (0..20000).map(|k| { let a = k as f64 * 0.002; Vector::new(a.sin(), a.cos(), (0.5 * a).sin()).normalize() }).collect();
        for (label, dirs) in [("random", &dirs_rand), ("coherent", &dirs_coh)] {
            let t = Instant::now();
            let mut s = 0.0;
            for d in dirs.iter() { s += hull.local_support_point(*d).x; }
            let brute = t.elapsed().as_nanos() as f64 / dirs.len() as f64;
            let t = Instant::now();
            for d in dirs.iter() { s += fast.local_support_point(*d).x; }
            let walk = t.elapsed().as_nanos() as f64 / dirs.len() as f64;
            println!("hull {nv:>5} verts {label:>8}: scan {brute:6.1} ns   walk {walk:6.1} ns   ({s:.0})");
        }
    }
}
