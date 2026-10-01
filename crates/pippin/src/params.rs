//! Runtime model parameters, for domain randomization and scene variation.
//!
//! Each call updates one element and keeps derived quantities consistent
//! (bounding radii, collision shapes, collision pairs, inverse inertia).

use crate::math::{Real, Vec3};
use crate::model::{GeomType, Model};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Param {
    /// Sliding friction coefficient (1 value).
    GeomFriction,
    /// Primitive size, same layout as MJCF `size` (3 values). Collision and
    /// rendering change; mass does not (set `BodyMass` explicitly).
    GeomSize,
    /// Color (4 values). Rendering only.
    GeomRgba,
    /// Collision bitmasks (1 value each). Setting both to 0 removes a geom
    /// from collision, e.g. to park unused objects.
    GeomContype,
    GeomConaffinity,
    /// Body mass (1 value). Inertia scales by the same factor, so the shape
    /// of the mass distribution is preserved.
    BodyMass,
    /// Joint damping per dof (1 value).
    DofDamping,
    /// Actuator gain (1 value).
    ActuatorGain,
    /// Actuator bias terms b0, b1, b2 (3 values).
    ActuatorBias,
    /// Body inertia about its center of mass, principal moments along the
    /// body frame axes (3 values).
    BodyInertia,
    /// Body center of mass in the body frame (3 values).
    BodyIpos,
    /// State used by reset (nq values, id 0). The kinematic reference
    /// configuration (MJCF `qpos0`, `ref`) is not affected.
    Qpos0,
}

impl Param {
    pub fn from_name(name: &str) -> Option<Param> {
        Some(match name {
            "geom_friction" => Param::GeomFriction,
            "geom_size" => Param::GeomSize,
            "geom_rgba" => Param::GeomRgba,
            "geom_contype" => Param::GeomContype,
            "geom_conaffinity" => Param::GeomConaffinity,
            "body_mass" => Param::BodyMass,
            "dof_damping" => Param::DofDamping,
            "actuator_gain" => Param::ActuatorGain,
            "actuator_bias" => Param::ActuatorBias,
            "body_inertia" => Param::BodyInertia,
            "body_ipos" => Param::BodyIpos,
            "qpos0" => Param::Qpos0,
            _ => return None,
        })
    }

    /// Values per element (`Qpos0` is nq wide; see [`Param::width_in`]).
    pub fn width(self) -> usize {
        match self {
            Param::GeomSize | Param::ActuatorBias | Param::BodyInertia | Param::BodyIpos => 3,
            Param::GeomRgba => 4,
            _ => 1,
        }
    }

    /// Values per element for model `m`.
    pub fn width_in(self, m: &Model) -> usize {
        if self == Param::Qpos0 { m.nq } else { self.width() }
    }

    /// Number of elements of this kind in the model.
    pub fn count(self, m: &Model) -> usize {
        match self {
            Param::GeomFriction | Param::GeomSize | Param::GeomRgba | Param::GeomContype | Param::GeomConaffinity => m.ngeom(),
            Param::BodyMass | Param::BodyInertia | Param::BodyIpos => m.nbody(),
            Param::Qpos0 => 1,
            Param::DofDamping => m.nv,
            Param::ActuatorGain | Param::ActuatorBias => m.nu,
        }
    }

    /// Whether changing this parameter affects physics (vs. rendering only).
    pub fn is_physical(self) -> bool {
        self != Param::GeomRgba
    }
}

impl Model {
    /// Read the current value of one element.
    pub fn get_param(&self, p: Param, id: usize) -> Vec<Real> {
        match p {
            Param::GeomFriction => vec![self.geom_friction[id]],
            Param::GeomSize => self.geom_size[id].0.to_vec(),
            Param::GeomRgba => self.geom_rgba[id].iter().map(|&x| x as Real).collect(),
            Param::GeomContype => vec![self.geom_contype[id] as Real],
            Param::GeomConaffinity => vec![self.geom_conaffinity[id] as Real],
            Param::BodyMass => vec![self.body_mass[id]],
            Param::DofDamping => vec![self.dof_damping[id]],
            Param::ActuatorGain => vec![self.actuator_gain[id]],
            Param::ActuatorBias => self.actuator_bias[id].to_vec(),
            Param::BodyInertia => (0..3).map(|k| self.body_inertia[id].0[4 * k]).collect(),
            Param::BodyIpos => self.body_ipos[id].0.to_vec(),
            Param::Qpos0 => self.qpos_reset.clone(),
        }
    }

    /// Set one element. Fails on out-of-range ids, wrong widths, or invalid values.
    pub fn set_param(&mut self, p: Param, id: usize, v: &[Real]) -> Result<(), String> {
        if id >= p.count(self) {
            return Err(format!("{p:?}: id {id} out of range (have {})", p.count(self)));
        }
        if v.len() != p.width_in(self) {
            return Err(format!("{p:?}: expected {} values, got {}", p.width_in(self), v.len()));
        }
        if v.iter().any(|x| !x.is_finite()) {
            return Err(format!("{p:?}: values must be finite"));
        }
        match p {
            Param::GeomFriction => self.geom_friction[id] = v[0].max(0.0),
            Param::GeomSize => {
                if matches!(self.geom_type[id], GeomType::Mesh | GeomType::Plane) {
                    return Err("geom_size applies to sphere, capsule, box, and cylinder geoms".into());
                }
                if v.iter().take(3).any(|&x| x < 0.0) {
                    return Err("geom_size must be non-negative".into());
                }
                self.geom_size[id] = Vec3::new(v[0], v[1], v[2]);
                self.geom_rbound[id] = self.rbound(id);
                self.geom_aabb[id] = self.local_aabb(id);
                self.geom_shape[id] = self.build_shape(id);
            }
            Param::GeomRgba => self.geom_rgba[id] = [v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32],
            Param::GeomContype | Param::GeomConaffinity => {
                let bits = v[0].max(0.0) as u32;
                if p == Param::GeomContype {
                    self.geom_contype[id] = bits;
                } else {
                    self.geom_conaffinity[id] = bits;
                }
                self.rebuild_collision_pairs();
            }
            Param::BodyMass => {
                if id == 0 {
                    return Err("the world body has no mass".into());
                }
                if v[0] <= 0.0 && self.body_jntnum[id] > 0 {
                    return Err("moving bodies need positive mass".into());
                }
                let scale = if self.body_mass[id] > 0.0 { v[0] / self.body_mass[id] } else { 1.0 };
                self.body_mass[id] = v[0];
                self.body_inertia[id] = self.body_inertia[id].scale(scale);
                crate::forward::set_const(self);
            }
            Param::DofDamping => self.dof_damping[id] = v[0].max(0.0),
            Param::ActuatorGain => self.actuator_gain[id] = v[0],
            Param::ActuatorBias => self.actuator_bias[id] = [v[0], v[1], v[2]],
            Param::BodyInertia => {
                if id == 0 || v.iter().any(|&x| x < 0.0) {
                    return Err("body_inertia needs a non-world body and non-negative moments".into());
                }
                self.body_inertia[id] = crate::math::Mat3::diag(Vec3::new(v[0], v[1], v[2]));
                crate::forward::set_const(self);
            }
            Param::BodyIpos => {
                if id == 0 {
                    return Err("the world body has no center of mass".into());
                }
                self.body_ipos[id] = Vec3::new(v[0], v[1], v[2]);
                crate::forward::set_const(self);
            }
            Param::Qpos0 => self.qpos_reset.copy_from_slice(v),
        }
        Ok(())
    }
}
