//! Print the contacts of a model at a keyframe (debug aid).
//! `cargo run --release --example contacts -- model.xml [key]`
fn main() {
    let path = std::env::args().nth(1).expect("model path");
    let m = pippin::mjcf::load_file(&path).expect("load");
    let mut d = pippin::Data::new(&m);
    if let Some(k) = std::env::args().nth(2).and_then(|k| m.key_id(&k)) {
        d.qpos.copy_from_slice(&m.key_qpos[k]);
    }
    pippin::forward(&m, &mut d);
    println!("{} contacts ({} candidate pairs)", d.contacts.len(), m.collision_pairs.len());
    for c in &d.contacts {
        let (g1, g2) = (c.geom[0], c.geom[1]);
        println!(
            "  {} [{}] - {} [{}]  depth {:+.5}",
            m.geom_names[g1], m.body_names[m.geom_body[g1]], m.geom_names[g2], m.body_names[m.geom_body[g2]], c.depth
        );
    }
}
