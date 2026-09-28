//! Rendered silhouettes and depths against analytic projections.

use pippin_env::{GeomVisual, Pose, RenderConfig, Renderer, Scene, Shape, ViewSource};
use pippin_render::MetalRenderer;

const W: usize = 128;
const H: usize = 128;
const FOVY: f32 = 60.0;

fn identity(pos: [f32; 3]) -> Pose {
    Pose { pos, mat: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] }
}

fn geom(shape: Shape, size: [f32; 3]) -> GeomVisual {
    GeomVisual { name: "g".into(), shape, size, rgba: [0.8, 0.2, 0.2, 1.0], body: 1, group: 0 }
}

/// Render one geom from a camera `dist` above it, looking down.
fn render(shape: Shape, size: [f32; 3], pose: Pose, dist: f32) -> (Vec<f32>, Vec<i32>) {
    let scene = Scene { geoms: vec![geom(shape, size)], meshes: vec![], cameras: vec![] };
    let cfg = RenderConfig {
        width: W,
        height: H,
        views: vec![ViewSource::look_at([0.0, 0.0, dist], [0.0, 0.0, 0.0], FOVY)],
        near: 0.01,
        far: 100.0,
    };
    let mut r = MetalRenderer::new(&scene, cfg).unwrap();
    let f = r.render(1, &[pose], &[]).unwrap();
    (f.depth.to_vec(), f.segmentation.to_vec())
}

/// Pixels per world unit at depth z for this camera.
fn px_per_unit(z: f32) -> f32 {
    let f = 1.0 / (0.5 * FOVY.to_radians()).tan();
    f / z * (H as f32 / 2.0)
}

fn coverage(seg: &[i32]) -> f32 {
    seg.iter().filter(|&&s| s == 0).count() as f32
}

#[test]
fn sphere_silhouette_and_depth() {
    let (r, d) = (0.2f32, 2.0f32);
    let (depth, seg) = render(Shape::Sphere, [r, 0.0, 0.0], identity([0.0; 3]), d);
    let center = (H / 2) * W + W / 2;
    assert_eq!(seg[center], 0);
    assert!((depth[center] - (d - r)).abs() < 2e-3, "center depth {}", depth[center]);
    // silhouette radius of a sphere: r / sqrt(d^2 - r^2) in normalized units
    let rad_px = r / (d * d - r * r).sqrt() * (1.0 / (0.5 * FOVY.to_radians()).tan()) * (H as f32 / 2.0);
    let expected = std::f32::consts::PI * rad_px * rad_px;
    let got = coverage(&seg);
    assert!((got - expected).abs() / expected < 0.05, "coverage {got} vs {expected}");
    // background is far and unlabeled
    assert_eq!(seg[0], -1);
    assert!((depth[0] - 100.0).abs() < 1e-3);
}

#[test]
fn box_silhouette_axis_aligned_and_rotated() {
    let (h, d) = (0.1f32, 1.5f32);
    let (depth, seg) = render(Shape::Box, [h, h, h], identity([0.0; 3]), d);
    let side = 2.0 * h * px_per_unit(d - h);
    let got = coverage(&seg);
    assert!((got - side * side).abs() / (side * side) < 0.05, "coverage {got} vs {}", side * side);
    assert!((depth[(H / 2) * W + W / 2] - (d - h)).abs() < 2e-3);

    // rotated 45 degrees about z: same area, diamond-shaped
    let c = std::f32::consts::FRAC_1_SQRT_2;
    let rot = Pose { pos: [0.0; 3], mat: [c, -c, 0.0, c, c, 0.0, 0.0, 0.0, 1.0] };
    let (_, seg) = render(Shape::Box, [h, h, h], rot, d);
    let got = coverage(&seg);
    assert!((got - side * side).abs() / (side * side) < 0.06, "rotated coverage {got} vs {}", side * side);
    // corner of the diamond reaches further along x than the square did
    let row = &seg[(H / 2) * W..(H / 2 + 1) * W];
    let width = row.iter().filter(|&&s| s == 0).count() as f32;
    assert!((width - side * std::f32::consts::SQRT_2).abs() < 3.0, "diamond width {width}");
}

#[test]
fn capsule_and_cylinder_lying_down() {
    let (r, hl, d) = (0.05f32, 0.2f32, 2.0f32);
    // rotate the z axis onto x
    let lie = Pose { pos: [0.0; 3], mat: [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0] };
    let (_, seg) = render(Shape::Capsule, [r, hl, 0.0], lie, d);
    let s = px_per_unit(d - r);
    let expected = (2.0 * r * 2.0 * hl + std::f32::consts::PI * r * r) * s * s;
    let got = coverage(&seg);
    assert!((got - expected).abs() / expected < 0.08, "capsule {got} vs {expected}");
    let (_, seg) = render(Shape::Cylinder, [r, hl, 0.0], lie, d);
    let expected = 2.0 * r * 2.0 * hl * s * s;
    let got = coverage(&seg);
    assert!((got - expected).abs() / expected < 0.08, "cylinder {got} vs {expected}");
}

#[test]
fn tilted_box_projection() {
    // cube tilted 30 degrees about x: top view is 2h x 2h(cos + sin)
    let (h, d) = (0.1f32, 1.5f32);
    let t = 30f32.to_radians();
    let (c, s) = (t.cos(), t.sin());
    let pose = Pose { pos: [0.0; 3], mat: [1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c] };
    let (_, seg) = render(Shape::Box, [h, h, h], pose, d);
    let col: usize = (0..H).filter(|&y| seg[y * W + W / 2] == 0).count();
    let row: usize = (0..W).filter(|&x| seg[(H / 2) * W + x] == 0).count();
    let scale = px_per_unit(d - 0.1);
    eprintln!("tilted box: row {row}px (expect ~{:.0}), col {col}px (expect ~{:.0})", 2.0 * h * scale, 2.0 * h * (c + s) * scale);
    assert!(col as f32 > 2.0 * h * (c + s) * px_per_unit(d) * 0.9);
    assert!(row as f32 > 2.0 * h * px_per_unit(d) * 0.9);
}
