//! Both physics backends, driven only through the `Physics` trait, must agree.

use pippin_env::{Field, Physics, PippinCpu, Pose};
use pippin_mujoco::MujocoPhysics;

fn drive(p: &mut dyn Physics, steps: usize) -> (Vec<f64>, Vec<Pose>) {
    let n = p.num_envs();
    let dims = p.dims();
    for t in 0..steps {
        let ctrl: Vec<f64> = (0..n * dims.nu).map(|i| ((t * 7 + i) as f64 * 0.13).sin()).collect();
        p.set(Field::Ctrl, 0..n, &ctrl);
        // step the two halves separately, as the async pipeline will
        p.step(0..n / 2, 1);
        p.step(n / 2..n, 1);
    }
    let mut q = vec![0.0; n * dims.nq];
    p.get(Field::Qpos, 0..n, &mut q);
    let mut g = vec![Pose::default(); n * dims.ngeom];
    let mut c = vec![Pose::default(); n * dims.ncam];
    p.poses(0..n, &mut g, &mut c);
    (q, g)
}

#[test]
fn pippin_and_mujoco_agree_through_trait() {
    for (file, steps) in [("box_drop.xml", 500), ("ant.xml", 100), ("cartpole.xml", 200)] {
        let path = format!("../../assets/{file}");
        let mut a = PippinCpu::from_file(&path, 4).unwrap();
        let mut b = MujocoPhysics::from_file(&path, 4).unwrap();
        assert_eq!(a.dims(), b.dims(), "{file}");
        assert_eq!(a.scene().geoms.len(), b.scene().geoms.len());
        let (qa, ga) = drive(&mut a, steps);
        let (qb, gb) = drive(&mut b, steps);
        let qerr = qa.iter().zip(&qb).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        let perr = ga.iter().zip(&gb).flat_map(|(x, y)| (0..3).map(move |i| (x.pos[i] - y.pos[i]).abs())).fold(0.0, f32::max);
        eprintln!("{file}: qpos err {qerr:.1e}, geom pose err {perr:.1e}");
        assert!(qerr < 1e-8, "{file}: {qerr}");
        assert!(perr < 1e-5, "{file}: {perr}");
    }
}
