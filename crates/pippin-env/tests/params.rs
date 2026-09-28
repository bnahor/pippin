//! Per-environment parameters: each env must behave according to its own values.

use pippin_env::{Appearance, Field, Param, Physics, PippinCpu};

fn sim(xml: &str, n: usize) -> PippinCpu {
    PippinCpu::new(pippin::mjcf::load_str(xml).unwrap(), n)
}

fn qpos(p: &PippinCpu) -> Vec<Vec<f64>> {
    let (n, nq) = (p.num_envs(), p.dims().nq);
    let mut q = vec![0.0; n * nq];
    p.get(Field::Qpos, 0..n, &mut q);
    q.chunks(nq).map(|c| c.to_vec()).collect()
}

const INCLINE: &str = r#"<mujoco><option timestep="0.002"/><worldbody>
    <geom name="floor" type="plane" size="5 5 .1" euler="0 25 0" friction="0.1"/>
    <body name="b" pos="0 0 0.1" euler="0 25 0"><freejoint/><geom name="box" type="box" size=".05 .05 .05" friction="0.1"/></body>
</worldbody></mujoco>"#;

#[test]
fn friction_per_env() {
    // tan(25 deg) = 0.47: mu = 0.1 slides, mu = 0.9 sticks (combined = max)
    let mut p = sim(INCLINE, 2);
    let (floor, boxg) = (0, 1);
    p.set_param(Param::GeomFriction, boxg, 1..2, &[0.9]).unwrap();
    p.set_param(Param::GeomFriction, floor, 1..2, &[0.9]).unwrap();
    p.step(0..2, 500);
    let q = qpos(&p);
    let slid = |q: &Vec<f64>| (q[0].powi(2) + q[1].powi(2)).sqrt();
    assert!(slid(&q[0]) > 0.1, "low friction env should slide: {}", slid(&q[0]));
    assert!(slid(&q[1]) < 0.01, "high friction env should stick: {}", slid(&q[1]));
    // base model untouched, env 0 still shares it
    let mut out = [0.0; 2];
    p.get_param(Param::GeomFriction, boxg, 0..2, &mut out).unwrap();
    assert_eq!(out, [0.1, 0.9]);
}

#[test]
fn size_and_parking_per_env() {
    let xml = r#"<mujoco><worldbody><geom type="plane" size="5 5 .1"/>
        <body pos="0 0 0.3"><freejoint/><geom name="ball" type="sphere" size=".05"/></body>
    </worldbody></mujoco>"#;
    let mut p = sim(xml, 3);
    p.set_param(Param::GeomSize, 1, 1..2, &[0.1, 0.0, 0.0]).unwrap();
    // env 2: remove the ball from collision entirely ("parked")
    p.set_param(Param::GeomContype, 1, 2..3, &[0.0]).unwrap();
    p.set_param(Param::GeomConaffinity, 1, 2..3, &[0.0]).unwrap();
    p.step(0..3, 1000);
    let q = qpos(&p);
    assert!((q[0][2] - 0.05).abs() < 2e-3, "r=0.05 rests at {}", q[0][2]);
    assert!((q[1][2] - 0.10).abs() < 2e-3, "r=0.10 rests at {}", q[1][2]);
    assert!(q[2][2] < -1.0, "parked ball should fall through: {}", q[2][2]);
    // renderer sees the size change as a per-env scale
    let mut ap = vec![Appearance::default(); 3 * 2];
    p.appearance(0..3, &mut ap);
    assert_eq!(ap[2 + 1].scale[0], 2.0);
    assert_eq!(ap[1].scale[0], 1.0);
}

#[test]
fn mass_per_env_changes_actuated_motion() {
    let xml = r#"<mujoco><option gravity="0 0 0"/><worldbody>
        <body name="link"><joint name="j" axis="0 0 1"/><geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/></body>
    </worldbody><actuator><motor joint="j"/></actuator></mujoco>"#;
    let mut p = sim(xml, 2);
    p.set_param(Param::BodyMass, 1, 1..2, &[4.0]).unwrap();
    p.set(Field::Ctrl, 0..2, &[1.0, 1.0]);
    p.step(0..2, 200);
    let q = qpos(&p);
    // same torque, 4x inertia -> 1/4 the angle
    let ratio = q[0][0] / q[1][0];
    assert!((ratio - 4.0).abs() < 1e-6, "angle ratio {ratio}");
}

#[test]
fn invalid_params_are_rejected_without_partial_updates() {
    let mut p = sim(INCLINE, 2);
    assert!(p.set_param(Param::GeomSize, 0, 0..2, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).is_err(), "plane has no size");
    assert!(p.set_param(Param::GeomFriction, 99, 0..1, &[1.0]).is_err());
    assert!(p.set_param(Param::BodyMass, 1, 0..1, &[-1.0]).is_err());
    let mut out = [0.0; 2];
    p.get_param(Param::GeomFriction, 1, 0..2, &mut out).unwrap();
    assert_eq!(out, [0.1, 0.1]);
}
