//! Compiled, immutable model. Flat structure-of-arrays so the same layout can be
//! uploaded to GPU buffers unchanged.

use crate::math::{Mat3, Quat, Real, Vec3};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JointType {
    Free,
    Ball,
    Slide,
    Hinge,
}

impl JointType {
    pub fn nq(self) -> usize {
        match self {
            JointType::Free => 7,
            JointType::Ball => 4,
            _ => 1,
        }
    }
    pub fn nv(self) -> usize {
        match self {
            JointType::Free => 6,
            JointType::Ball => 3,
            _ => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeomType {
    Plane,
    Sphere,
    Capsule,
    Box,
    Cylinder,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Integrator {
    /// Semi-implicit Euler with implicit joint damping (MuJoCo "Euler").
    Euler,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverKind {
    /// Convex primal Newton over accelerations with pyramidal friction cones.
    Newton,
    /// Soft projected Gauss-Seidel on velocities with split-impulse relaxation.
    Pgs,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolverOptions {
    pub kind: SolverKind,
    pub iterations: usize,
    /// PGS: stop once no row changes its constraint velocity by more than this.
    /// Newton: stop once the relative gradient norm falls below this.
    pub tolerance: Real,
    /// Contacts are generated this far before geoms touch.
    pub contact_margin: Real,

    // ---- Newton: MuJoCo-style soft constraint parameters ----
    /// (timeconst, dampratio) of the constraint reference dynamics.
    pub solref: [Real; 2],
    /// (dmin, dmax, width, midpoint, power) of the impedance curve.
    pub solimp: [Real; 5],
    /// Ratio of frictional to normal constraint impedance (> 1 reduces slip).
    pub impratio: Real,

    // ---- PGS ----
    /// Contact spring frequency in Hz (0 => derived from the timestep).
    pub contact_hertz: Real,
    pub contact_damping_ratio: Real,
    /// Penetration tolerated before the spring engages.
    pub contact_slop: Real,
}

impl Default for SolverOptions {
    fn default() -> Self {
        SolverOptions {
            kind: SolverKind::Newton,
            iterations: 30,
            tolerance: 1e-8,
            contact_margin: 1e-3,
            solref: [0.02, 1.0],
            solimp: [0.9, 0.95, 0.001, 0.5, 2.0],
            impratio: 1.0,
            contact_hertz: 0.0,
            contact_damping_ratio: 1.0,
            contact_slop: 5e-4,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Model {
    pub timestep: Real,
    pub gravity: Vec3,
    pub integrator: Integrator,
    pub solver: SolverOptions,

    pub nq: usize,
    pub nv: usize,
    pub nu: usize,

    // ---- bodies (body 0 is the world) ----
    pub body_names: Vec<String>,
    pub body_parent: Vec<usize>,
    pub body_pos: Vec<Vec3>,
    pub body_quat: Vec<Quat>,
    pub body_jntadr: Vec<usize>,
    pub body_jntnum: Vec<usize>,
    pub body_dofadr: Vec<usize>,
    pub body_dofnum: Vec<usize>,
    /// Last dof on the path from this body to the world, or usize::MAX.
    pub body_lastdof: Vec<usize>,
    pub body_mass: Vec<Real>,
    /// Center of mass in the body frame.
    pub body_ipos: Vec<Vec3>,
    /// Inertia about the com, body-frame axes.
    pub body_inertia: Vec<Mat3>,
    /// First ancestor body (or self) that is attached to the world by a free
    /// joint chain root; used for parent filtering.
    pub body_weldid: Vec<usize>,

    // ---- joints ----
    pub jnt_names: Vec<String>,
    pub jnt_type: Vec<JointType>,
    pub jnt_body: Vec<usize>,
    pub jnt_qposadr: Vec<usize>,
    pub jnt_dofadr: Vec<usize>,
    pub jnt_pos: Vec<Vec3>,
    pub jnt_axis: Vec<Vec3>,
    pub jnt_limited: Vec<bool>,
    pub jnt_range: Vec<[Real; 2]>,
    pub jnt_stiffness: Vec<Real>,
    pub jnt_springref: Vec<Real>,

    // ---- dofs ----
    pub dof_body: Vec<usize>,
    pub dof_jnt: Vec<usize>,
    /// Parent dof in the kinematic tree, or usize::MAX.
    pub dof_parent: Vec<usize>,
    pub dof_damping: Vec<Real>,
    pub dof_armature: Vec<Real>,

    // ---- geoms ----
    pub geom_names: Vec<String>,
    pub geom_type: Vec<GeomType>,
    pub geom_body: Vec<usize>,
    pub geom_pos: Vec<Vec3>,
    pub geom_quat: Vec<Quat>,
    pub geom_size: Vec<Vec3>,
    pub geom_friction: Vec<Real>,
    pub geom_contype: Vec<u32>,
    pub geom_conaffinity: Vec<u32>,
    pub geom_rgba: Vec<[f32; 4]>,
    /// Bounding sphere radius about the geom center (inf for planes).
    pub geom_rbound: Vec<Real>,

    // ---- cameras (look along -z, y up, as in MuJoCo) ----
    pub cam_names: Vec<String>,
    pub cam_body: Vec<usize>,
    pub cam_pos: Vec<Vec3>,
    pub cam_quat: Vec<Quat>,
    /// Vertical field of view in degrees.
    pub cam_fovy: Vec<Real>,

    // ---- actuators (joint transmission only for now) ----
    pub actuator_names: Vec<String>,
    pub actuator_joint: Vec<usize>,
    pub actuator_gear: Vec<Real>,
    pub actuator_gain: Vec<Real>,
    /// bias = b0 + b1 * length + b2 * velocity
    pub actuator_bias: Vec<[Real; 3]>,
    pub actuator_ctrllimited: Vec<bool>,
    pub actuator_ctrlrange: Vec<[Real; 2]>,
    pub actuator_forcelimited: Vec<bool>,
    pub actuator_forcerange: Vec<[Real; 2]>,

    /// Inverse inertia at qpos0, (translational, rotational) per body; used to
    /// scale constraint softness like MuJoCo's `body_invweight0`.
    pub body_invweight: Vec<[Real; 2]>,
    /// Inverse joint-space inertia diagonal at qpos0 (MuJoCo `dof_invweight0`).
    pub dof_invweight: Vec<Real>,

    /// Reference configuration.
    pub qpos0: Vec<Real>,
    /// Geom pairs that may collide (static filtering already applied).
    pub collision_pairs: Vec<(usize, usize)>,
}

impl Model {
    pub fn nbody(&self) -> usize {
        self.body_parent.len()
    }
    pub fn njnt(&self) -> usize {
        self.jnt_type.len()
    }
    pub fn ngeom(&self) -> usize {
        self.geom_type.len()
    }

    pub fn body_id(&self, name: &str) -> Option<usize> {
        self.body_names.iter().position(|n| n == name)
    }
    pub fn joint_id(&self, name: &str) -> Option<usize> {
        self.jnt_names.iter().position(|n| n == name)
    }
    pub fn geom_id(&self, name: &str) -> Option<usize> {
        self.geom_names.iter().position(|n| n == name)
    }
    pub fn actuator_id(&self, name: &str) -> Option<usize> {
        self.actuator_names.iter().position(|n| n == name)
    }

    /// Finalize derived quantities. Called by loaders after filling raw fields.
    pub(crate) fn compile(&mut self) {
        let nbody = self.nbody();

        // dof tree
        self.body_lastdof = vec![usize::MAX; nbody];
        self.dof_parent = vec![usize::MAX; self.nv];
        for b in 1..nbody {
            let mut last = self.body_lastdof[self.body_parent[b]];
            for k in 0..self.body_dofnum[b] {
                let d = self.body_dofadr[b] + k;
                self.dof_parent[d] = last;
                last = d;
            }
            self.body_lastdof[b] = last;
        }

        // weld ids: bodies rigidly attached share an id
        self.body_weldid = (0..nbody).collect();
        for b in 1..nbody {
            if self.body_jntnum[b] == 0 {
                self.body_weldid[b] = self.body_weldid[self.body_parent[b]];
            }
        }

        self.geom_rbound = (0..self.ngeom())
            .map(|g| {
                let s = self.geom_size[g];
                match self.geom_type[g] {
                    GeomType::Plane => Real::INFINITY,
                    GeomType::Sphere => s[0],
                    GeomType::Capsule => s[0] + s[1],
                    GeomType::Box => s.norm(),
                    GeomType::Cylinder => (s[0] * s[0] + s[1] * s[1]).sqrt(),
                }
            })
            .collect();

        // static collision filtering
        self.collision_pairs.clear();
        for g1 in 0..self.ngeom() {
            for g2 in (g1 + 1)..self.ngeom() {
                if self.can_collide(g1, g2) {
                    self.collision_pairs.push((g1, g2));
                }
            }
        }

        if self.solver.contact_hertz <= 0.0 {
            // Stiffest spring the integrator can resolve stably, with margin.
            self.solver.contact_hertz = 0.25 / self.timestep;
        }
    }

    fn can_collide(&self, g1: usize, g2: usize) -> bool {
        let (b1, b2) = (self.geom_body[g1], self.geom_body[g2]);
        let (w1, w2) = (self.body_weldid[b1], self.body_weldid[b2]);
        if w1 == w2 {
            return false;
        }
        let masks = (self.geom_contype[g1] & self.geom_conaffinity[g2]) != 0
            || (self.geom_contype[g2] & self.geom_conaffinity[g1]) != 0;
        if !masks {
            return false;
        }
        // parent filtering, except against the world
        let p1 = self.body_weldid[self.body_parent[w1]];
        let p2 = self.body_weldid[self.body_parent[w2]];
        if (p1 == w2 && w2 != 0) || (p2 == w1 && w1 != 0) {
            return false;
        }
        crate::collision::pair_supported(self.geom_type[g1], self.geom_type[g2])
    }
}
