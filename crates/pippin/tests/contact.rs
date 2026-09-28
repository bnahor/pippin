//! Physical-behaviour tests for contact, friction, and joint limits, checked
//! against analytic answers.

use pippin::math::{Real, Vec3};
use pippin::model::SolverKind;
use pippin::{mjcf, step, Data, Model};

/// Run `check` once per constraint solver.
fn each_solver(xml: &str, check: impl Fn(&Model, Data)) {
    for kind in [SolverKind::Newton, SolverKind::Pgs] {
        let mut m = mjcf::load_str(xml).expect("model loads");
        m.solver.kind = kind;
        eprintln!("solver {kind:?}");
        let d = Data::new(&m);
        check(&m, d);
    }
}

fn run(m: &Model, d: &mut Data, seconds: Real) {
    let n = (seconds / m.timestep).round() as usize;
    for _ in 0..n {
        step(m, d);
    }
}

fn body_pos(m: &Model, d: &Data, name: &str) -> Vec3 {
    d.xpos[m.body_id(name).unwrap()]
}

#[test]
fn box_rests_on_plane() {
    each_solver(
        r#"<mujoco><worldbody>
            <geom type="plane" size="5 5 .1"/>
            <body name="b" pos="0 0 0.3" euler="3 5 0"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
        </worldbody></mujoco>"#,
        |m, mut d| {
            run(m, &mut d, 3.0);
            let p = body_pos(m, &d, "b");
            assert!((p[2] - 0.1).abs() < 2e-3, "rest height {}", p[2]);
            assert!(p[0].abs() < 1e-2 && p[1].abs() < 1e-2, "drifted to {p:?}");
            let speed: Real = d.qvel.iter().map(|v| v * v).sum::<Real>().sqrt();
            assert!(speed < 1e-3, "still moving: {speed}");
        },
    );
}

#[test]
fn sphere_and_capsule_rest() {
    each_solver(
        r#"<mujoco><worldbody>
            <geom type="plane" size="5 5 .1"/>
            <body name="s" pos="0 0 0.5"><freejoint/><geom type="sphere" size=".05"/></body>
            <body name="c" pos="1 0 0.5" euler="90 0 0"><freejoint/><geom type="capsule" size=".04 .2"/></body>
        </worldbody></mujoco>"#,
        |m, mut d| {
            run(m, &mut d, 3.0);
            assert!((body_pos(m, &d, "s")[2] - 0.05).abs() < 2e-3);
            assert!((body_pos(m, &d, "c")[2] - 0.04).abs() < 2e-3);
        },
    );
}

#[test]
fn box_stack_is_stable() {
    let mut xml = String::from(r#"<mujoco><option timestep="0.002"/><worldbody><geom type="plane" size="5 5 .1"/>"#);
    for i in 0..5 {
        xml += &format!(
            r#"<body name="b{i}" pos="0 0 {}"><freejoint/><geom type="box" size=".05 .05 .05"/></body>"#,
            0.05 + 0.1 * i as Real + 0.001 * i as Real
        );
    }
    xml += "</worldbody></mujoco>";
    each_solver(&xml, |m, mut d| {
        run(m, &mut d, 5.0);
        let top = body_pos(m, &d, "b4");
        assert!(top[0].abs() < 5e-3 && top[1].abs() < 5e-3, "top box drifted: {top:?}");
        assert!((top[2] - 0.45).abs() < 1e-2, "top box height {}", top[2]);
    });
}

fn incline_xml(deg: Real, mu: Real) -> String {
    format!(
        r#"<mujoco><option timestep="0.002"/><worldbody>
            <geom type="plane" size="5 5 .1" euler="0 {deg} 0" friction="{mu}"/>
            <body name="b" pos="0 0 0.1" euler="0 {deg} 0"><freejoint/><geom type="box" size=".05 .05 .05" friction="{mu}"/></body>
        </worldbody></mujoco>"#
    )
}

#[test]
fn friction_holds_below_critical_angle() {
    // tan(20 deg) = 0.36 < mu = 0.5
    each_solver(&incline_xml(20.0, 0.5), |m, mut d| {
        run(m, &mut d, 0.5);
        let p0 = body_pos(m, &d, "b");
        run(m, &mut d, 1.0);
        let p1 = body_pos(m, &d, "b");
        assert!((p1 - p0).norm() < 2e-3, "box crept {:.4} m", (p1 - p0).norm());
    });
}

#[test]
fn friction_slides_with_analytic_acceleration() {
    // a = g (sin t - mu cos t) along the slope
    let (deg, mu): (Real, Real) = (35.0, 0.3);
    each_solver(&incline_xml(deg, mu), |m, mut d| {
        run(m, &mut d, 0.3);
        let v0 = Vec3::new(d.qvel[0], d.qvel[1], d.qvel[2]).norm();
        run(m, &mut d, 0.5);
        let v1 = Vec3::new(d.qvel[0], d.qvel[1], d.qvel[2]).norm();
        let t = deg.to_radians();
        let expected = 9.81 * (t.sin() - mu * t.cos());
        let measured = (v1 - v0) / 0.5;
        assert!(
            (measured - expected).abs() / expected < 0.03,
            "a = {measured:.3}, expected {expected:.3}"
        );
    });
}

#[test]
fn joint_limit_is_respected() {
    each_solver(
        r#"<mujoco><option timestep="0.002"/><worldbody>
            <body pos="0 0 1"><joint name="h" axis="0 1 0" range="-30 30"/>
              <geom type="capsule" fromto="0 0 0 0.5 0 0" size="0.03" contype="0" conaffinity="0"/></body>
        </worldbody></mujoco>"#,
        |m, mut d| {
            let mut qmax: Real = 0.0;
            for _ in 0..2000 {
                step(m, &mut d);
                qmax = qmax.max(d.qpos[0]);
            }
            let lim = 30f64.to_radians();
            // soft limits overshoot on impact; MuJoCo overshoots this scene by 0.037 rad
            assert!(qmax < lim + 0.05, "exceeded limit: {qmax} > {lim}");
            assert!((d.qpos[0] - lim).abs() < 0.01, "should rest at limit, at {}", d.qpos[0]);
        },
    );
}

#[test]
fn sphere_on_box_and_capsule_on_box() {
    each_solver(
        r#"<mujoco><worldbody>
            <geom type="box" size="1 1 .1" pos="0 0 -.1"/>
            <body name="s" pos="0 0 0.3"><freejoint/><geom type="sphere" size=".05"/></body>
            <body name="c" pos="0.5 0 0.3" euler="0 90 0"><freejoint/><geom type="capsule" size=".03 .1"/></body>
            <body name="bx" pos="-0.5 0 0.3" euler="0 0 30"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
        </worldbody></mujoco>"#,
        |m, mut d| {
            run(m, &mut d, 3.0);
            assert!((body_pos(m, &d, "s")[2] - 0.05).abs() < 2e-3);
            assert!((body_pos(m, &d, "c")[2] - 0.03).abs() < 2e-3);
            assert!((body_pos(m, &d, "bx")[2] - 0.05).abs() < 2e-3);
        },
    );
}

#[test]
fn rollout_matches_step_loop() {
    use pippin::batch::Field;
    use pippin::Batch;
    for path in ["../../assets/cartpole.xml", "../../assets/tumble.xml"] {
        let m = mjcf::load_file(path).unwrap();
        let (nu, nq, n, t) = (m.nu, m.nq, 3, 50);
        let ctrl: Vec<Real> = (0..n * t * nu).map(|i| (i as Real * 0.37).sin()).collect();
        let mut a = Batch::new(m.clone(), n);
        let mut out = vec![0.0; n * t * nq];
        a.rollout(&ctrl, t, &mut out);
        let mut b = Batch::new(m, n);
        for k in 0..t {
            let c: Vec<Real> = (0..n).flat_map(|e| ctrl[(e * t + k) * nu..(e * t + k + 1) * nu].to_vec()).collect();
            b.set(Field::Ctrl, &c);
            b.step(1);
        }
        let mut q = vec![0.0; n * nq];
        b.get(Field::Qpos, &mut q);
        for e in 0..n {
            assert_eq!(&out[(e * t + t - 1) * nq..(e * t + t) * nq], &q[e * nq..(e + 1) * nq], "{path}");
        }
    }
}
