//! Engine-agnostic interfaces that decouple physics from rendering.
//!
//! A [`Physics`] backend owns a batch of environments. It steps any contiguous
//! range of them (so environment groups can be pipelined against rendering and
//! inference), exposes state as flat arrays, and exports a static
//! [`Scene`] plus per-environment [`Pose`]s of every geom and camera. Renderers
//! consume only `Scene` + poses, so any physics engine works with any renderer.

use std::ops::Range;

pub mod pippin_cpu;

pub use pippin_cpu::PippinCpu;

/// State arrays a backend exposes, shaped (envs, width).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field {
    Qpos,
    Qvel,
    Ctrl,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dims {
    pub nq: usize,
    pub nv: usize,
    pub nu: usize,
    pub ngeom: usize,
    pub ncam: usize,
}

impl Dims {
    pub fn width(&self, f: Field) -> usize {
        match f {
            Field::Qpos => self.nq,
            Field::Qvel => self.nv,
            Field::Ctrl => self.nu,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    /// Infinite plane through the origin with normal +z; `size` x/y give a
    /// visual half-extent (0 = infinite).
    Plane,
    Sphere,
    /// Radius, half-length along z.
    Capsule,
    /// Half-extents.
    Box,
    /// Radius, half-length along z.
    Cylinder,
    Ellipsoid,
    /// Index into [`Scene::meshes`].
    Mesh(usize),
}

#[derive(Clone, Debug, PartialEq)]
pub struct GeomVisual {
    pub name: String,
    pub shape: Shape,
    pub size: [f32; 3],
    pub rgba: [f32; 4],
    /// Body the geom is attached to (for segmentation by body).
    pub body: usize,
    /// Visual group (MuJoCo convention: 0-2 visible by default).
    pub group: i32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    pub name: String,
    pub vertices: Vec<[f32; 3]>,
    pub triangles: Vec<[u32; 3]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CameraDesc {
    pub name: String,
    /// Vertical field of view, degrees.
    pub fovy: f32,
}

/// Everything static a renderer needs. Exported once per model.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Scene {
    pub geoms: Vec<GeomVisual>,
    pub meshes: Vec<Mesh>,
    pub cameras: Vec<CameraDesc>,
}

/// World pose: position and row-major rotation matrix. Cameras look along
/// their local -z with +y up.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pose {
    pub pos: [f32; 3],
    pub mat: [f32; 9],
}

/// A batch of environments simulated by one physics engine.
pub trait Physics: Send {
    fn name(&self) -> &str;
    fn num_envs(&self) -> usize;
    fn dims(&self) -> Dims;
    fn scene(&self) -> Scene;

    /// Advance environments `envs` by `nstep` steps.
    fn step(&mut self, envs: Range<usize>, nstep: usize);
    /// Reset the listed environments to their initial state.
    fn reset(&mut self, envs: &[usize]);

    /// Copy `field` for `envs` into `out`, shaped (envs.len(), width).
    fn get(&self, field: Field, envs: Range<usize>, out: &mut [f64]);
    /// Overwrite `field` for `envs` from `src`, shaped (envs.len(), width).
    fn set(&mut self, field: Field, envs: Range<usize>, src: &[f64]);

    /// World poses for `envs`: geoms into `geoms` (envs x ngeom) and cameras
    /// into `cams` (envs x ncam). Poses reflect the most recent step.
    fn poses(&self, envs: Range<usize>, geoms: &mut [Pose], cams: &mut [Pose]);
}

/// Check buffer sizes shared by all backends.
pub fn check_len(what: &str, got: usize, envs: &Range<usize>, width: usize) {
    assert_eq!(got, envs.len() * width, "{what}: buffer has {got} elements, expected {} x {width}", envs.len());
}
