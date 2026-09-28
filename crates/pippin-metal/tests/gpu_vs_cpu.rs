//! The GPU (f32) backend must track the CPU (f64) reference engine.

use pippin::{mjcf, step, Data};
use pippin_metal::MetalSim;

/// Max |qpos| difference between GPU and CPU after `seconds`, with sinusoidal controls.
fn max_err(path: &str, seconds: f64) -> f64 {
    let m = mjcf::load_file(path).unwrap();
    let n = 4;
    let mut gpu = MetalSim::new(m.clone(), n).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut d = Data::new(&m);
    let steps = (seconds / m.timestep).round() as usize;
    let mut err: f64 = 0.0;
    for t in 0..steps {
        for u in 0..m.nu {
            let c = ((t as f64) * 0.05 + u as f64).sin();
            d.ctrl[u] = c;
            for e in 0..n {
                gpu.ctrl_mut()[e * m.nu + u] = c as f32;
            }
        }
        step(&m, &mut d);
        gpu.step(1).unwrap();
        for e in 0..n {
            for k in 0..m.nq {
                err = err.max((gpu.qpos()[e * m.nq + k] as f64 - d.qpos[k]).abs());
            }
        }
    }
    err
}

#[test]
fn contact_free_models_track_cpu() {
    for (name, tol) in [("pendulum", 1e-4), ("cartpole", 1e-3), ("arm3d", 1e-3), ("tumble", 1e-3)] {
        let e = max_err(&format!("../../assets/{name}.xml"), 0.5);
        eprintln!("{name}: gpu vs cpu max qpos err {e:.2e}");
        assert!(e < tol, "{name}: {e}");
    }
}

#[test]
fn contact_models_track_cpu() {
    for (name, tol) in [("box_drop", 1e-3), ("ant", 1e-2)] {
        let e = max_err(&format!("../../assets/{name}.xml"), 0.5);
        eprintln!("{name}: gpu vs cpu max qpos err {e:.2e}");
        assert!(e < tol, "{name}: {e}");
    }
}
