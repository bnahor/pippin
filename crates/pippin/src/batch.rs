//! Many environments sharing one model, stepped in parallel.

use rayon::prelude::*;

use crate::data::Data;
use crate::forward;
use crate::math::Real;
use crate::model::Model;

pub struct Batch {
    pub model: Model,
    pub envs: Vec<Data>,
}

impl Batch {
    pub fn new(model: Model, n: usize) -> Batch {
        let proto = Data::new(&model);
        Batch { envs: vec![proto; n], model }
    }

    pub fn len(&self) -> usize {
        self.envs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.envs.is_empty()
    }

    /// Step every environment `nstep` times.
    pub fn step(&mut self, nstep: usize) {
        let m = &self.model;
        let chunk = self.chunk_len();
        self.envs.par_chunks_mut(chunk).for_each(|envs| {
            for _ in 0..nstep {
                for d in envs.iter_mut() {
                    forward::step(m, d);
                }
            }
        });
    }

    /// Recompute derived quantities (poses, contacts) without advancing time.
    pub fn forward(&mut self) {
        let m = &self.model;
        let chunk = self.chunk_len();
        self.envs.par_chunks_mut(chunk).for_each(|envs| envs.iter_mut().for_each(|d| forward::forward(m, d)));
    }

    /// A few chunks per thread: coarse enough to amortize scheduling, fine
    /// enough for work stealing to balance performance and efficiency cores.
    fn chunk_len(&self) -> usize {
        (self.envs.len() / (4 * rayon::current_num_threads())).max(1)
    }

    pub fn reset(&mut self, ids: &[usize]) {
        for &i in ids {
            self.envs[i].reset(&self.model);
        }
    }

    pub fn get(&self, field: Field, out: &mut [Real]) {
        let w = field.width(&self.model);
        assert_eq!(out.len(), w * self.len(), "output buffer has the wrong size");
        for (d, o) in self.envs.iter().zip(out.chunks_mut(w)) {
            o.copy_from_slice(field.slice(d));
        }
    }

    pub fn set(&mut self, field: Field, src: &[Real]) {
        let w = field.width(&self.model);
        assert_eq!(src.len(), w * self.len(), "input buffer has the wrong size");
        for (d, s) in self.envs.iter_mut().zip(src.chunks(w)) {
            field.slice_mut(d).copy_from_slice(s);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field {
    Qpos,
    Qvel,
    Ctrl,
    QfrcApplied,
}

impl Field {
    pub fn width(self, m: &Model) -> usize {
        match self {
            Field::Qpos => m.nq,
            Field::Qvel | Field::QfrcApplied => m.nv,
            Field::Ctrl => m.nu,
        }
    }
    fn slice(self, d: &Data) -> &[Real] {
        match self {
            Field::Qpos => &d.qpos,
            Field::Qvel => &d.qvel,
            Field::Ctrl => &d.ctrl,
            Field::QfrcApplied => &d.qfrc_applied,
        }
    }
    fn slice_mut(self, d: &mut Data) -> &mut [Real] {
        match self {
            Field::Qpos => &mut d.qpos,
            Field::Qvel => &mut d.qvel,
            Field::Ctrl => &mut d.ctrl,
            Field::QfrcApplied => &mut d.qfrc_applied,
        }
    }
}
