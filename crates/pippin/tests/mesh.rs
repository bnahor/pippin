//! Mesh and cylinder collision (general convex narrow phase).

use pippin::math::{Real, Vec3};
use pippin::{mjcf, step, Data, Model};

fn run(m: &Model, d: &mut Data, seconds: Real) {
    for _ in 0..(seconds / m.timestep).round() as usize {
        step(m, d);
    }
}

fn pos(m: &Model, d: &Data, body: &str) -> Vec3 {
    d.xpos[m.body_id(body).unwrap()]
}

fn speed(d: &Data) -> Real {
    d.qvel.iter().map(|v| v * v).sum::<Real>().sqrt()
}

const CUBE: &str = "-0.05 -0.05 -0.05  0.05 -0.05 -0.05  -0.05 0.05 -0.05  0.05 0.05 -0.05 \
                    -0.05 -0.05 0.05  0.05 -0.05 0.05  -0.05 0.05 0.05  0.05 0.05 0.05";

#[test]
fn mesh_cube_matches_box_mass_and_rests() {
    let xml = format!(
        r#"<mujoco><asset><mesh name="cube" vertex="{CUBE}"/></asset><worldbody>
            <geom type="plane" size="5 5 .1"/>
            <body name="m" pos="0 0 0.3" euler="4 7 0"><freejoint/><geom type="mesh" mesh="cube"/></body>
            <body name="b" pos="1 0 0.3" euler="4 7 0"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
        </worldbody></mujoco>"#
    );
    let m = mjcf::load_str(&xml).unwrap();
    let (bm, bb) = (m.body_id("m").unwrap(), m.body_id("b").unwrap());
    assert!((m.body_mass[bm] - m.body_mass[bb]).abs() < 1e-12, "mass {} vs {}", m.body_mass[bm], m.body_mass[bb]);
    for k in 0..9 {
        assert!((m.body_inertia[bm].0[k] - m.body_inertia[bb].0[k]).abs() < 1e-12);
    }
    let mut d = Data::new(&m);
    run(&m, &mut d, 3.0);
    let (pm, pb) = (pos(&m, &d, "m"), pos(&m, &d, "b"));
    assert!((pm[2] - pb[2]).abs() < 1e-3, "mesh rests at {} vs box {}", pm[2], pb[2]);
    assert!((pm[2] - 0.05).abs() < 2e-3);
    assert!(speed(&d) < 1e-3, "still moving: {}", speed(&d));
}

#[test]
fn mesh_cubes_stack() {
    let mut xml = format!(r#"<mujoco><asset><mesh name="cube" vertex="{CUBE}"/></asset><worldbody><geom type="plane" size="5 5 .1"/>"#);
    for i in 0..4 {
        xml += &format!(
            r#"<body name="c{i}" pos="0 0 {}" euler="0 0 {}"><freejoint/><geom type="mesh" mesh="cube"/></body>"#,
            0.05 + 0.101 * i as Real,
            9 * i
        );
    }
    xml += "</worldbody></mujoco>";
    let m = mjcf::load_str(&xml).unwrap();
    let mut d = Data::new(&m);
    run(&m, &mut d, 5.0);
    let top = pos(&m, &d, "c3");
    assert!(top[0].abs() < 5e-3 && top[1].abs() < 5e-3, "top drifted {top:?}");
    assert!((top[2] - 0.35).abs() < 1e-2, "top height {}", top[2]);
}

#[test]
fn cylinders_rest_upright_and_on_their_side() {
    let m = mjcf::load_str(
        r#"<mujoco><worldbody>
            <geom type="box" size="1 1 .1" pos="0 0 -.1"/>
            <body name="up" pos="0 0 0.3"><freejoint/><geom type="cylinder" size=".04 .06"/></body>
            <body name="side" pos="0.5 0 0.3" euler="90 0 0"><freejoint/><geom type="cylinder" size=".04 .06"/></body>
            <body name="onup" pos="0 0 0.5"><freejoint/><geom type="cylinder" size=".03 .03"/></body>
        </worldbody></mujoco>"#,
    )
    .unwrap();
    let mut d = Data::new(&m);
    run(&m, &mut d, 3.0);
    assert!((pos(&m, &d, "up")[2] - 0.06).abs() < 2e-3, "upright at {}", pos(&m, &d, "up")[2]);
    assert!((pos(&m, &d, "side")[2] - 0.04).abs() < 2e-3, "side at {}", pos(&m, &d, "side")[2]);
    // small cylinder stacked on the upright one
    assert!((pos(&m, &d, "onup")[2] - (0.12 + 0.03)).abs() < 3e-3, "stacked at {}", pos(&m, &d, "onup")[2]);
}

#[test]
fn mesh_from_obj_file_with_scale() {
    // unit tetrahedron-like wedge; scaled x2 must double every extent
    let dir = std::env::temp_dir().join("pippin_mesh_test");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("wedge.obj"),
        "v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nv 0 0 0.1\nv 0.1 0.1 0\nf 1 3 2\nf 2 3 5\nf 1 2 4\nf 1 4 3\nf 2 5 4\nf 3 4 5\n",
    )
    .unwrap();
    let xml = r#"<mujoco><asset><mesh file="wedge.obj" scale="2 2 2"/></asset><worldbody>
        <geom type="plane" size="5 5 .1"/>
        <body name="w" pos="0 0 0.5"><freejoint/><geom type="mesh" mesh="wedge"/></body>
    </worldbody></mujoco>"#;
    let m = mjcf::load_str_in(xml, &dir).unwrap();
    let hull = &m.mesh_hull[0];
    let max_x = hull.vertices.iter().map(|v| v[0]).fold(Real::MIN, Real::max);
    assert!((max_x - 0.2).abs() < 1e-6, "max_x {max_x}"); // OBJ coordinates are f32
    // volume of the scaled hull: square pyramid, base 0.2 x 0.2, apex height 0.2
    let (vol, _, _) = pippin::mesh::mass_properties(hull);
    assert!((vol - 0.2 * 0.2 * 0.2 / 3.0).abs() < 1e-8, "vol {vol}");
    let mut d = Data::new(&m);
    run(&m, &mut d, 3.0);
    assert!(speed(&d) < 1e-3, "wedge did not settle");
    // lowest hull point rests on the floor
    let b = m.body_id("w").unwrap();
    let lowest = hull.vertices.iter().map(|v| (d.xpos[b] + d.xmat[b].mul_vec(*v))[2]).fold(Real::MAX, Real::min);
    assert!(lowest.abs() < 2e-3, "lowest point at {lowest}");
}
