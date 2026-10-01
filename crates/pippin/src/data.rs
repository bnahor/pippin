//! Mutable simulation state for one environment.

use crate::collision::Contact;
use crate::math::{Mat3, Quat, Real, Spatial, SpatialInertia, Vec3};
use crate::model::Model;

#[derive(Clone, Debug)]
pub struct Data {
    pub time: Real,
    pub qpos: Vec<Real>,
    pub qvel: Vec<Real>,
    pub ctrl: Vec<Real>,
    /// User-applied generalized forces.
    pub qfrc_applied: Vec<Real>,

    // ---- kinematics ----
    pub xpos: Vec<Vec3>,
    pub xquat: Vec<Quat>,
    pub xmat: Vec<Mat3>,
    /// Body com in world frame.
    pub xipos: Vec<Vec3>,
    pub xanchor: Vec<Vec3>,
    pub xaxis: Vec<Vec3>,
    pub geom_xpos: Vec<Vec3>,
    pub geom_xmat: Vec<Mat3>,

    // ---- dynamics ----
    pub cdof: Vec<Spatial>,
    pub cdof_dot: Vec<Spatial>,
    pub cinert: Vec<SpatialInertia>,
    pub crb: Vec<SpatialInertia>,
    pub cvel: Vec<Spatial>,
    /// Dense joint-space inertia (nv x nv, row-major), including armature.
    pub qm: Vec<Real>,
    /// Cholesky factor of qm + h*damping.
    pub qm_chol: Vec<Real>,
    pub qfrc_bias: Vec<Real>,
    pub qfrc_passive: Vec<Real>,
    pub qfrc_actuator: Vec<Real>,
    pub qfrc_constraint: Vec<Real>,
    pub qacc: Vec<Real>,
    pub actuator_force: Vec<Real>,
    /// Whether each actuator's force hit its force range this step.
    pub actuator_clamped: Vec<bool>,

    // ---- contacts / constraints ----
    pub contacts: Vec<Contact>,
    pub(crate) scratch: Scratch,
}

impl Data {
    /// Iterations used by the Newton solver on the last step (diagnostics).
    pub fn solver_iterations(&self) -> usize {
        self.scratch.newton_iters
    }
}

/// Solver workspace, kept to avoid per-step allocation.
#[derive(Clone, Debug, Default)]
pub(crate) struct Scratch {
    pub hits: Vec<crate::collision::Hit>,
    pub rows: Vec<crate::solver::Row>,
    pub jac: Vec<Real>,
    pub minv_jt: Vec<Real>,
    pub lambda: Vec<Real>,
    pub prev_contacts: Vec<Contact>,
    pub prev_lambda: Vec<[Real; 3]>,
    pub tmp_nv: Vec<Real>,
    pub v: Vec<Real>,
    pub v_pos: Vec<Real>,
    pub smooth: Vec<Real>,
    pub nrows: Vec<crate::newton::NRow>,
    pub damp_h: Vec<Real>,
    pub newton_buf: Vec<Real>,
    pub newton_iters: usize,
    pub island: crate::newton::IslandBuf,
    pub aabb: Vec<(Vec3, Vec3)>,
    pub cacc: Vec<Spatial>,
    pub cfrc: Vec<Spatial>,
}

impl Data {
    pub fn new(m: &Model) -> Data {
        let nb = m.nbody();
        let ng = m.ngeom();
        let nj = m.njnt();
        let nv = m.nv;
        Data {
            time: 0.0,
            qpos: m.qpos_reset.clone(),
            qvel: vec![0.0; nv],
            ctrl: vec![0.0; m.nu],
            qfrc_applied: vec![0.0; nv],
            xpos: vec![Vec3::ZERO; nb],
            xquat: vec![Quat::IDENTITY; nb],
            xmat: vec![Mat3::IDENTITY; nb],
            xipos: vec![Vec3::ZERO; nb],
            xanchor: vec![Vec3::ZERO; nj],
            xaxis: vec![Vec3::ZERO; nj],
            geom_xpos: vec![Vec3::ZERO; ng],
            geom_xmat: vec![Mat3::IDENTITY; ng],
            cdof: vec![Spatial::ZERO; nv],
            cdof_dot: vec![Spatial::ZERO; nv],
            cinert: vec![SpatialInertia::default(); nb],
            crb: vec![SpatialInertia::default(); nb],
            cvel: vec![Spatial::ZERO; nb],
            qm: vec![0.0; nv * nv],
            qm_chol: vec![0.0; nv * nv],
            qfrc_bias: vec![0.0; nv],
            qfrc_passive: vec![0.0; nv],
            qfrc_actuator: vec![0.0; nv],
            qfrc_constraint: vec![0.0; nv],
            qacc: vec![0.0; nv],
            actuator_force: vec![0.0; m.nu],
            actuator_clamped: vec![false; m.nu],
            contacts: vec![],
            scratch: Scratch::default(),
        }
    }

    /// Reset to the reference configuration.
    pub fn reset(&mut self, m: &Model) {
        self.time = 0.0;
        self.qpos.copy_from_slice(&m.qpos_reset);
        self.qvel.fill(0.0);
        self.ctrl.fill(0.0);
        self.qfrc_applied.fill(0.0);
        self.contacts.clear();
        self.scratch.prev_contacts.clear();
        self.scratch.prev_lambda.clear();
    }
}
