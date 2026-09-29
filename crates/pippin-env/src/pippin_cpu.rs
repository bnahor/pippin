//! Physics backend: Pippin's CPU engine, stepping environments across cores.

use std::ops::Range;
use std::sync::Arc;

use pippin::math::{Mat3, Quat, Real, Vec3};
use pippin::model::GeomType;
use pippin::params::Param;
use pippin::{forward, Data, Model};
use rayon::prelude::*;

use crate::{check_len, Appearance, CameraDesc, Dims, Field, GeomVisual, Mesh, Physics, Pose, Scene, Shape};

pub struct PippinCpu {
    /// Base model; defines the scene and dimensions shared by all envs.
    pub model: Model,
    /// Per-environment model. Environments share the base model until a
    /// parameter is changed for them (copy on write; meshes stay shared).
    pub models: Vec<Arc<Model>>,
    pub envs: Vec<Data>,
}

impl PippinCpu {
    pub fn new(model: Model, n: usize) -> PippinCpu {
        pippin::threads::init();
        let mut proto = Data::new(&model);
        forward::kinematics(&model, &mut proto);
        let shared = Arc::new(model.clone());
        PippinCpu { envs: vec![proto; n], models: vec![shared; n], model }
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
                    GeomType::Mesh => Shape::Mesh(m.geom_dataid[g]),
                },
                size: m.geom_size[g].0.map(|x| x as f32),
                rgba: m.geom_rgba[g],
                body: m.geom_body[g],
                group: m.geom_group[g],
            })
            .collect();
        let cameras = (0..m.cam_body.len())
            .map(|c| CameraDesc { name: m.cam_names[c].clone(), fovy: m.cam_fovy[c] as f32 })
            .collect();
        let meshes = m
            .mesh
            .iter()
            .zip(&m.mesh_names)
            .map(|(t, name)| Mesh {
                name: name.clone(),
                vertices: t.vertices.iter().map(|v| v.0.map(|x| x as f32)).collect(),
                triangles: t.triangles.clone(),
            })
            .collect();
        Scene { bodies: m.body_names.clone(), geoms, meshes, cameras }
    }

    fn step(&mut self, envs: Range<usize>, nstep: usize) {
        let chunk = (envs.len() / (4 * rayon::current_num_threads())).max(1);
        let models = &self.models[envs.clone()];
        self.envs[envs].par_chunks_mut(chunk).zip(models.par_chunks(chunk)).for_each(|(ds, ms)| {
            for _ in 0..nstep {
                for (d, m) in ds.iter_mut().zip(ms) {
                    forward::step(m, d);
                }
            }
            // keep poses in sync with the new state for rendering
            for (d, m) in ds.iter_mut().zip(ms) {
                forward::kinematics(m, d);
            }
        });
    }

    fn reset(&mut self, envs: &[usize]) {
        for &i in envs {
            let d = &mut self.envs[i];
            d.reset(&self.models[i]);
            forward::kinematics(&self.models[i], d);
        }
    }

    fn set_param(&mut self, p: Param, id: usize, envs: Range<usize>, values: &[f64]) -> Result<(), String> {
        let w = p.width_in(&self.model);
        check_len("set_param", values.len(), &envs, w);
        // validate against the base model first so no env is left half-updated
        self.model.clone().set_param(p, id, &values[..w])?;
        for (i, v) in envs.zip(values.chunks(w)) {
            Arc::make_mut(&mut self.models[i]).set_param(p, id, v)?;
        }
        Ok(())
    }

    fn get_param(&self, p: Param, id: usize, envs: Range<usize>, out: &mut [f64]) -> Result<(), String> {
        let w = p.width_in(&self.model);
        check_len("get_param", out.len(), &envs, w);
        if id >= p.count(&self.model) {
            return Err(format!("{p:?}: id {id} out of range"));
        }
        for (i, o) in envs.zip(out.chunks_mut(w)) {
            o.copy_from_slice(&self.models[i].get_param(p, id));
        }
        Ok(())
    }

    fn appearance(&self, envs: Range<usize>, out: &mut [Appearance]) {
        let ng = self.model.ngeom();
        check_len("appearance", out.len(), &envs, ng);
        for (k, i) in envs.enumerate() {
            let m = &self.models[i];
            for g in 0..ng {
                let base = self.model.geom_size[g];
                let size = m.geom_size[g];
                let scale = std::array::from_fn(|a| if base[a] > 0.0 { (size[a] / base[a]) as f32 } else { 1.0 });
                out[k * ng + g] = Appearance { rgba: m.geom_rgba[g], scale, pad: 0.0 };
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
            o.copy_from_slice(field(d, f));
        }
    }

    fn set(&mut self, f: Field, envs: Range<usize>, src: &[f64]) {
        let w = self.dims().width(f);
        check_len("set", src.len(), &envs, w);
        if w == 0 {
            return;
        }
        let models = &self.models[envs.clone()];
        for ((d, s), m) in self.envs[envs].iter_mut().zip(src.chunks(w)).zip(models) {
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
