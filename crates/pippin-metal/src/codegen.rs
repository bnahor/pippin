//! Generates the model-specific header spliced into `engine.metal`: sizes as
//! macros and model constants as `constant` arrays, so the Metal compiler sees
//! compile-time loop bounds and keeps model data in constant memory.

use std::fmt::Write;

use pippin::math::{Quat, Real, Vec3};
use pippin::model::{GeomType, JointType, Model};

/// Upper bound on contacts one geom pair can produce.
fn pair_max_contacts(a: GeomType, b: GeomType) -> usize {
    use GeomType::*;
    let (a, b) = if (a as u8) <= (b as u8) { (a, b) } else { (b, a) };
    match (a, b) {
        (Plane, Capsule) => 2,
        (Plane, Box) | (Plane, Cylinder) | (Box, Box) => 4,
        (Capsule, Capsule) | (Capsule, Box) => 3,
        _ => 1,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Maximum contacts kept per environment per step.
    pub max_contacts: usize,
    /// Maximum constraint rows per environment per step.
    pub max_rows: usize,
}

impl Limits {
    pub fn for_model(m: &Model) -> Limits {
        let possible: usize = m
            .collision_pairs
            .iter()
            .map(|&(a, b)| pair_max_contacts(m.geom_type[a], m.geom_type[b]))
            .sum();
        let max_contacts = possible.clamp(1, 32);
        let nlimited = (0..m.njnt()).filter(|&j| m.jnt_limited[j]).count();
        let max_rows = (4 * max_contacts).min(64) + 2 * nlimited;
        Limits { max_contacts, max_rows: max_rows.max(1) }
    }
}

fn f(x: Real) -> String {
    let x = x as f32;
    if x.is_infinite() {
        if x > 0.0 { "INFINITY".into() } else { "(-INFINITY)".into() }
    } else {
        format!("{x:?}f")
    }
}

fn v3(v: Vec3) -> String {
    format!("float3({}, {}, {})", f(v[0]), f(v[1]), f(v[2]))
}

fn q4(q: Quat) -> String {
    format!("float4({}, {}, {}, {})", f(q.0[0]), f(q.0[1]), f(q.0[2]), f(q.0[3]))
}

/// `constant <ty> name[n] = {...};`, padded with `pad` so no array is empty.
fn array<T>(out: &mut String, ty: &str, name: &str, items: &[T], fmt: impl Fn(&T) -> String, pad: &str) {
    let mut body: Vec<String> = items.iter().map(fmt).collect();
    if body.is_empty() {
        body.push(pad.to_string());
    }
    writeln!(out, "constant {ty} {name}[{}] = {{{}}};", body.len(), body.join(", ")).unwrap();
}

fn idx(i: usize) -> String {
    if i == usize::MAX { "-1".into() } else { i.to_string() }
}

pub fn header(m: &Model, lim: Limits) -> String {
    let mut s = String::new();
    let opt = &m.solver;
    let defines: [(&str, String); 22] = [
        ("NQ", m.nq.to_string()),
        ("NV", m.nv.to_string()),
        ("NU", m.nu.to_string()),
        ("NU_ALLOC", m.nu.max(1).to_string()),
        ("NBODY", m.nbody().to_string()),
        ("NJNT", m.njnt().to_string()),
        ("NGEOM", m.ngeom().max(1).to_string()),
        ("NPAIR", m.collision_pairs.len().to_string()),
        ("MAXCON", lim.max_contacts.to_string()),
        ("MAXROWS", lim.max_rows.to_string()),
        ("ITERATIONS", opt.iterations.max(1).to_string()),
        ("TOLERANCE", f(opt.tolerance.max(1e-6))),
        ("TIMESTEP", f(m.timestep)),
        ("MARGIN", f(opt.contact_margin)),
        ("SOLREF0", f(opt.solref[0])),
        ("SOLREF1", f(opt.solref[1])),
        ("SOLIMP0", f(opt.solimp[0])),
        ("SOLIMP1", f(opt.solimp[1])),
        ("SOLIMP2", f(opt.solimp[2])),
        ("SOLIMP3", f(opt.solimp[3])),
        ("SOLIMP4", f(opt.solimp[4])),
        ("IMPRATIO", f(opt.impratio)),
    ];
    for (k, v) in defines {
        writeln!(s, "#define {k} {v}").unwrap();
    }
    let damped = m.dof_damping.iter().any(|&b| b != 0.0);
    writeln!(s, "#define DAMPED {}", damped as i32).unwrap();
    writeln!(s, "constant float3 GRAVITY = {};", v3(m.gravity)).unwrap();

    let i = |x: &usize| x.to_string();
    array(&mut s, "int", "body_parent", &m.body_parent, i, "0");
    array(&mut s, "int", "body_jntadr", &m.body_jntadr, i, "0");
    array(&mut s, "int", "body_jntnum", &m.body_jntnum, i, "0");
    array(&mut s, "int", "body_dofadr", &m.body_dofadr, i, "0");
    array(&mut s, "int", "body_dofnum", &m.body_dofnum, i, "0");
    array(&mut s, "int", "body_lastdof", &m.body_lastdof, |x| idx(*x), "-1");
    array(&mut s, "float3", "body_pos", &m.body_pos, |x| v3(*x), "float3(0)");
    array(&mut s, "float4", "body_quat", &m.body_quat, |x| q4(*x), "float4(1,0,0,0)");
    array(&mut s, "float", "body_mass", &m.body_mass, |x| f(*x), "0");
    array(&mut s, "float3", "body_ipos", &m.body_ipos, |x| v3(*x), "float3(0)");
    array(
        &mut s,
        "float3x3",
        "body_inertia",
        &m.body_inertia,
        |x| format!("float3x3({}, {}, {})", v3(x.col(0)), v3(x.col(1)), v3(x.col(2))),
        "float3x3(0)",
    );
    array(&mut s, "float", "body_invweight", &m.body_invweight, |x| f(x[0]), "0");

    let jt = |t: &JointType| {
        match t {
            JointType::Free => "0",
            JointType::Ball => "1",
            JointType::Slide => "2",
            JointType::Hinge => "3",
        }
        .to_string()
    };
    array(&mut s, "int", "jnt_type", &m.jnt_type, jt, "0");
    array(&mut s, "int", "jnt_qposadr", &m.jnt_qposadr, i, "0");
    array(&mut s, "int", "jnt_dofadr", &m.jnt_dofadr, i, "0");
    array(&mut s, "float3", "jnt_pos", &m.jnt_pos, |x| v3(*x), "float3(0)");
    array(&mut s, "float3", "jnt_axis", &m.jnt_axis, |x| v3(*x), "float3(0)");
    let limited: Vec<bool> =
        (0..m.njnt()).map(|j| m.jnt_limited[j] && matches!(m.jnt_type[j], JointType::Hinge | JointType::Slide)).collect();
    array(&mut s, "bool", "jnt_limited", &limited, |x| x.to_string(), "false");
    array(&mut s, "float2", "jnt_range", &m.jnt_range, |r| format!("float2({}, {})", f(r[0]), f(r[1])), "float2(0)");
    array(&mut s, "float", "jnt_stiffness", &m.jnt_stiffness, |x| f(*x), "0");
    array(&mut s, "float", "jnt_springref", &m.jnt_springref, |x| f(*x), "0");

    array(&mut s, "int", "dof_body", &m.dof_body, i, "0");
    array(&mut s, "int", "dof_parent", &m.dof_parent, |x| idx(*x), "-1");
    array(&mut s, "float", "dof_damping", &m.dof_damping, |x| f(*x), "0");
    array(&mut s, "float", "dof_armature", &m.dof_armature, |x| f(*x), "0");
    array(&mut s, "float", "dof_invweight", &m.dof_invweight, |x| f(*x), "0");

    array(&mut s, "int", "geom_type", &m.geom_type, |t| (*t as u8).to_string(), "0");
    array(&mut s, "int", "geom_body", &m.geom_body, i, "0");
    array(&mut s, "float3", "geom_pos", &m.geom_pos, |x| v3(*x), "float3(0)");
    array(&mut s, "float4", "geom_quat", &m.geom_quat, |x| q4(*x), "float4(1,0,0,0)");
    array(&mut s, "float3", "geom_size", &m.geom_size, |x| v3(*x), "float3(0)");
    array(&mut s, "float", "geom_friction", &m.geom_friction, |x| f(*x), "0");
    array(&mut s, "float", "geom_rbound", &m.geom_rbound, |x| f(*x), "0");
    array(&mut s, "int2", "pair_geom", &m.collision_pairs, |(a, b)| format!("int2({a}, {b})"), "int2(0)");

    array(&mut s, "int", "act_joint", &m.actuator_joint, i, "0");
    array(&mut s, "float", "act_gear", &m.actuator_gear, |x| f(*x), "0");
    array(&mut s, "float", "act_gain", &m.actuator_gain, |x| f(*x), "0");
    array(&mut s, "float3", "act_bias", &m.actuator_bias, |b| format!("float3({}, {}, {})", f(b[0]), f(b[1]), f(b[2])), "float3(0)");
    array(&mut s, "bool", "act_ctrllimited", &m.actuator_ctrllimited, |x| x.to_string(), "false");
    array(&mut s, "float2", "act_ctrlrange", &m.actuator_ctrlrange, |r| format!("float2({}, {})", f(r[0]), f(r[1])), "float2(0)");
    array(&mut s, "bool", "act_forcelimited", &m.actuator_forcelimited, |x| x.to_string(), "false");
    array(&mut s, "float2", "act_forcerange", &m.actuator_forcerange, |r| format!("float2({}, {})", f(r[0]), f(r[1])), "float2(0)");
    array(&mut s, "float", "qpos0", &m.qpos0, |x| f(*x), "0");
    s
}

/// Full Metal source for a model.
pub fn source(m: &Model, lim: Limits) -> String {
    include_str!("engine.metal").replace("// PIPPIN_MODEL", &header(m, lim))
}
