//! MuJoCo physics backend: one `mjModel`, one `mjData` per environment,
//! stepped in parallel across CPU cores.

pub mod sys;

use std::ffi::{CStr, CString};
use std::ops::Range;
use std::os::raw::c_char;

use pippin_env::{check_len, CameraDesc, Dims, Field, GeomVisual, Mesh, Physics, Pose, Scene, Shape};
use rayon::prelude::*;

#[derive(Debug)]
pub struct MujocoError(pub String);

impl std::fmt::Display for MujocoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MuJoCo: {}", self.0)
    }
}
impl std::error::Error for MujocoError {}

/// Owned mjData. MuJoCo allows concurrent use of distinct mjData with a
/// shared read-only mjModel.
struct DataPtr(*mut sys::mjData);
unsafe impl Send for DataPtr {}
unsafe impl Sync for DataPtr {}

pub struct MujocoPhysics {
    m: *mut sys::mjModel,
    envs: Vec<DataPtr>,
}

unsafe impl Send for MujocoPhysics {}
unsafe impl Sync for MujocoPhysics {}

impl Drop for MujocoPhysics {
    fn drop(&mut self) {
        unsafe {
            for d in &self.envs {
                sys::mj_deleteData(d.0);
            }
            sys::mj_deleteModel(self.m);
        }
    }
}

/// Copy `n` elements starting at `ptr + offset` (MuJoCo arrays).
unsafe fn arr<'a, T>(ptr: *const T, offset: usize, n: usize) -> &'a [T] {
    if n == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(ptr.add(offset), n)
    }
}

impl MujocoPhysics {
    pub fn from_file(path: &str, n: usize) -> Result<MujocoPhysics, MujocoError> {
        pippin_env::init_threads();
        let cpath = CString::new(path).map_err(|e| MujocoError(e.to_string()))?;
        let mut err = [0 as c_char; 1000];
        let m = unsafe { sys::mj_loadXML(cpath.as_ptr(), std::ptr::null(), err.as_mut_ptr(), err.len() as i32) };
        if m.is_null() {
            let msg = unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy().into_owned();
            return Err(MujocoError(msg));
        }
        let envs = (0..n)
            .map(|_| unsafe {
                let d = sys::mj_makeData(m);
                sys::mj_forward(m, d);
                DataPtr(d)
            })
            .collect();
        Ok(MujocoPhysics { m, envs })
    }

    fn model(&self) -> &sys::mjModel {
        unsafe { &*self.m }
    }

    fn name_of(&self, obj: sys::mjtObj, id: usize) -> String {
        let p = unsafe { sys::mj_id2name(self.m, obj as i32, id as i32) };
        if p.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        }
    }
}

fn field_ptr(d: &sys::mjData, f: Field) -> *mut f64 {
    match f {
        Field::Qpos => d.qpos,
        Field::Qvel => d.qvel,
        Field::Ctrl => d.ctrl,
    }
}

fn pose(pos: &[f64], mat: &[f64]) -> Pose {
    Pose { pos: [pos[0] as f32, pos[1] as f32, pos[2] as f32], mat: std::array::from_fn(|i| mat[i] as f32) }
}

impl Physics for MujocoPhysics {
    fn name(&self) -> &str {
        "mujoco"
    }

    fn num_envs(&self) -> usize {
        self.envs.len()
    }

    fn dims(&self) -> Dims {
        let m = self.model();
        Dims { nq: m.nq as usize, nv: m.nv as usize, nu: m.nu as usize, ngeom: m.ngeom as usize, ncam: m.ncam as usize }
    }

    fn scene(&self) -> Scene {
        let m = self.model();
        let ng = m.ngeom as usize;
        let geoms = (0..ng)
            .map(|g| unsafe {
                let size = arr(m.geom_size, 3 * g, 3);
                let rgba = arr(m.geom_rgba, 4 * g, 4);
                // mjtGeom: PLANE 0, HFIELD 1, SPHERE 2, CAPSULE 3, ELLIPSOID 4, CYLINDER 5, BOX 6, MESH 7
                let shape = match *m.geom_type.add(g) {
                    0 => Shape::Plane,
                    2 => Shape::Sphere,
                    3 => Shape::Capsule,
                    4 => Shape::Ellipsoid,
                    5 => Shape::Cylinder,
                    6 => Shape::Box,
                    7 => Shape::Mesh(*m.geom_dataid.add(g) as usize),
                    _ => Shape::Box, // heightfields/SDFs: placeholder
                };
                GeomVisual {
                    name: self.name_of(sys::mjtObj_mjOBJ_GEOM, g),
                    shape,
                    size: [size[0] as f32, size[1] as f32, size[2] as f32],
                    rgba: [rgba[0], rgba[1], rgba[2], rgba[3]],
                    body: *m.geom_bodyid.add(g) as usize,
                    group: *m.geom_group.add(g),
                }
            })
            .collect();
        let meshes = (0..m.nmesh as usize)
            .map(|k| unsafe {
                let va = *m.mesh_vertadr.add(k) as usize;
                let vn = *m.mesh_vertnum.add(k) as usize;
                let fa = *m.mesh_faceadr.add(k) as usize;
                let fnum = *m.mesh_facenum.add(k) as usize;
                let v = arr(m.mesh_vert, 3 * va, 3 * vn);
                let f = arr(m.mesh_face, 3 * fa, 3 * fnum);
                Mesh {
                    name: self.name_of(sys::mjtObj_mjOBJ_MESH, k),
                    vertices: v.chunks(3).map(|c| [c[0], c[1], c[2]]).collect(),
                    triangles: f.chunks(3).map(|c| [c[0] as u32, c[1] as u32, c[2] as u32]).collect(),
                }
            })
            .collect();
        let cameras = (0..m.ncam as usize)
            .map(|c| CameraDesc {
                name: self.name_of(sys::mjtObj_mjOBJ_CAMERA, c),
                fovy: unsafe { *m.cam_fovy.add(c) } as f32,
            })
            .collect();
        Scene { geoms, meshes, cameras }
    }

    fn step(&mut self, envs: Range<usize>, nstep: usize) {
        let m = self.m as usize; // raw pointers are not Sync; share the address
        let chunk = (envs.len() / (4 * rayon::current_num_threads())).max(1);
        self.envs[envs].par_chunks(chunk).for_each(|ds| {
            let m = m as *const sys::mjModel;
            for _ in 0..nstep {
                for d in ds {
                    unsafe { sys::mj_step(m, d.0) };
                }
            }
            // refresh poses for rendering
            for d in ds {
                unsafe {
                    sys::mj_kinematics(m, d.0);
                    sys::mj_camlight(m, d.0);
                }
            }
        });
    }

    fn reset(&mut self, envs: &[usize]) {
        for &i in envs {
            unsafe {
                sys::mj_resetData(self.m, self.envs[i].0);
                sys::mj_forward(self.m, self.envs[i].0);
            }
        }
    }

    fn get(&self, f: Field, envs: Range<usize>, out: &mut [f64]) {
        let w = self.dims().width(f);
        check_len("get", out.len(), &envs, w);
        if w == 0 {
            return;
        }
        for (d, o) in self.envs[envs].iter().zip(out.chunks_mut(w)) {
            o.copy_from_slice(unsafe { arr(field_ptr(&*d.0, f), 0, w) });
        }
    }

    fn set(&mut self, f: Field, envs: Range<usize>, src: &[f64]) {
        let w = self.dims().width(f);
        check_len("set", src.len(), &envs, w);
        if w == 0 {
            return;
        }
        for (d, s) in self.envs[envs].iter().zip(src.chunks(w)) {
            unsafe {
                std::slice::from_raw_parts_mut(field_ptr(&*d.0, f), w).copy_from_slice(s);
                if f == Field::Qpos {
                    sys::mj_kinematics(self.m, d.0);
                    sys::mj_camlight(self.m, d.0);
                }
            }
        }
    }

    fn poses(&self, envs: Range<usize>, geoms: &mut [Pose], cams: &mut [Pose]) {
        let dims = self.dims();
        let (ng, nc) = (dims.ngeom, dims.ncam);
        check_len("poses(geoms)", geoms.len(), &envs, ng);
        check_len("poses(cams)", cams.len(), &envs, nc);
        for (k, d) in self.envs[envs].iter().enumerate() {
            let d = unsafe { &*d.0 };
            for g in 0..ng {
                geoms[k * ng + g] = unsafe { pose(arr(d.geom_xpos, 3 * g, 3), arr(d.geom_xmat, 9 * g, 9)) };
            }
            for c in 0..nc {
                cams[k * nc + c] = unsafe { pose(arr(d.cam_xpos, 3 * c, 3), arr(d.cam_xmat, 9 * c, 9)) };
            }
        }
    }
}
