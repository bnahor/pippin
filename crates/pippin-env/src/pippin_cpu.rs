//! Physics backend: Pippin's CPU engine, stepping environments across cores.

use std::ops::Range;

use pippin::math::{Mat3, Quat, Real, Vec3};
use pippin::model::GeomType;
use pippin::{forward, Data, Model};
use rayon::prelude::*;

use crate::{check_len, CameraDesc, Dims, Field, GeomVisual, Physics, Pose, Scene, Shape};

pub struct PippinCpu {
    pub model: Model,
    pub envs: Vec<Data>,
}

impl PippinCpu {
    pub fn new(model: Model, n: usize) -> PippinCpu {
        let mut proto = Data::new(&model);
        forward::kinematics(&model, &mut proto);
        PippinCpu { envs: vec![proto; n], model }
    }

    pub fn from_file(path: &str, n: usize) -> Result<PippinCpu, pippin::mjcf::MjcfError> {
        Ok(Self::new(pippin::mjcf::load_file(path)?, n))
    }
}

fn pose(pos: Vec3, mat: &Mat3) -> Pose {
    Pose { pos: pos.0.map(|x| x as f32), mat: mat.0.map(|x| x as f32) }
}

fn field<'a>(d: &'a Data, f: Field) -> &'a [Real] {
    match f {
        Field::Qpos => &d.qpos,
        Field::Qvel => &d.qvel,
        Field::Ctrl => &d.ctrl,
    }
}

impl Physics for PippinCpu {
    fn name(&self) -> &str {
        "pippin-cpu"
    }

    fn num_envs(&self) -> usize {
        self.envs.len()
    }

    fn dims(&self) -> Dims {
        let m = &self.model;
        Dims { nq: m.nq, nv: m.nv, nu: m.nu, ngeom: m.ngeom(), ncam: m.cam_body.len() }
    }

    fn scene(&self) -> Scene {
        let m = &self.model;
        let geoms = (0..m.ngeom())
            .map(|g| GeomVisual {
                name: m.geom_names[g].clone(),
                shape: match m.geom_type[g] {
                    GeomType::Plane => Shape::Plane,
                    GeomType::Sphere => Shape::Sphere,
                    GeomType::Capsule => Shape::Capsule,
                    GeomType::Box => Shape::Box,
                    GeomType::Cylinder => Shape::Cylinder,
                },
                size: m.geom_size[g].0.map(|x| x as f32),
                rgba: m.geom_rgba[g],
                body: m.geom_body[g],
                group: 0,
            })
            .collect();
        let cameras = (0..m.cam_body.len())
            .map(|c| CameraDesc { name: m.cam_names[c].clone(), fovy: m.cam_fovy[c] as f32 })
            .collect();
        Scene { geoms, meshes: vec![], cameras }
    }

    fn step(&mut self, envs: Range<usize>, nstep: usize) {
        let m = &self.model;
        let chunk = (envs.len() / (4 * rayon::current_num_threads())).max(1);
        self.envs[envs].par_chunks_mut(chunk).for_each(|ds| {
            for _ in 0..nstep {
                for d in ds.iter_mut() {
                    forward::step(m, d);
                }
            }
            // keep poses in sync with the new state for rendering
            for d in ds.iter_mut() {
                forward::kinematics(m, d);
            }
        });
    }

    fn reset(&mut self, envs: &[usize]) {
        for &i in envs {
            let d = &mut self.envs[i];
            d.reset(&self.model);
            forward::kinematics(&self.model, d);
        }
    }

    fn get(&self, f: Field, envs: Range<usize>, out: &mut [f64]) {
        let w = self.dims().width(f);
        check_len("get", out.len(), &envs, w);
        if w == 0 {
            return;
        }
        for (d, o) in self.envs[envs].iter().zip(out.chunks_mut(w)) {
            o.copy_from_slice(field(d, f));
        }
    }

    fn set(&mut self, f: Field, envs: Range<usize>, src: &[f64]) {
        let w = self.dims().width(f);
        check_len("set", src.len(), &envs, w);
        if w == 0 {
            return;
        }
        let m = &self.model;
        for (d, s) in self.envs[envs].iter_mut().zip(src.chunks(w)) {
            match f {
                Field::Qpos => d.qpos.copy_from_slice(s),
                Field::Qvel => d.qvel.copy_from_slice(s),
                Field::Ctrl => d.ctrl.copy_from_slice(s),
            }
            if f == Field::Qpos {
                forward::kinematics(m, d);
            }
        }
    }

    fn poses(&self, envs: Range<usize>, geoms: &mut [Pose], cams: &mut [Pose]) {
        let m = &self.model;
        let (ng, nc) = (m.ngeom(), m.cam_body.len());
        check_len("poses(geoms)", geoms.len(), &envs, ng);
        check_len("poses(cams)", cams.len(), &envs, nc);
        for (k, d) in self.envs[envs].iter().enumerate() {
            for g in 0..ng {
                geoms[k * ng + g] = pose(d.geom_xpos[g], &d.geom_xmat[g]);
            }
            for c in 0..nc {
                let b = m.cam_body[c];
                let pos = d.xpos[b] + d.xmat[b].mul_vec(m.cam_pos[c]);
                let mat = d.xmat[b].mul_mat(&Quat::to_mat(m.cam_quat[c]));
                cams[k * nc + c] = pose(pos, &mat);
            }
        }
    }
}
