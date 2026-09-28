//! MJCF (MuJoCo XML) loader. Supports the subset used by typical robot and
//! manipulation scenes; unsupported elements are ignored with a warning list.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::path::Path;

use roxmltree::{Document, Node};

use crate::math::{Mat3, Quat, Real, Vec3};
use crate::model::{GeomType, Integrator, JointType, Model, SolverKind, SolverOptions};

#[derive(Debug, thiserror::Error)]
pub enum MjcfError {
    #[error("xml parse error: {0}")]
    Xml(#[from] roxmltree::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid model: {0}")]
    Invalid(String),
}

fn invalid<T>(msg: impl Into<String>) -> Result<T, MjcfError> {
    Err(MjcfError::Invalid(msg.into()))
}

type Attrs = HashMap<String, String>;

/// Default classes, fully resolved (each class already includes its parents).
#[derive(Default, Clone)]
struct DefaultClass {
    by_tag: HashMap<String, Attrs>,
}

struct Ctx {
    degrees: bool,
    eulerseq: [u8; 3],
    autolimits: bool,
    defaults: HashMap<String, DefaultClass>,
    pub warnings: Vec<String>,
}

impl Ctx {
    fn get<'a>(&'a self, node: &'a Node, class: &str, name: &str) -> Option<&'a str> {
        if let Some(v) = node.attribute(name) {
            return Some(v);
        }
        let tag = node.tag_name().name();
        let class = node.attribute("class").unwrap_or(class);
        let dc = self.defaults.get(class)?;
        if let Some(v) = dc.by_tag.get(tag).and_then(|a| a.get(name)) {
            return Some(v.as_str());
        }
        // actuator shortcuts share the "general" defaults
        if matches!(tag, "motor" | "position" | "velocity") {
            if let Some(v) = dc.by_tag.get("general").and_then(|a| a.get(name)) {
                return Some(v.as_str());
            }
        }
        None
    }

    fn floats(&self, node: &Node, class: &str, name: &str) -> Result<Option<Vec<Real>>, MjcfError> {
        match self.get(node, class, name) {
            None => Ok(None),
            Some(s) => parse_floats(s).map(Some),
        }
    }

    fn float(&self, node: &Node, class: &str, name: &str, default: Real) -> Result<Real, MjcfError> {
        Ok(self.floats(node, class, name)?.and_then(|v| v.first().copied()).unwrap_or(default))
    }

    fn vec3(&self, node: &Node, class: &str, name: &str, default: Vec3) -> Result<Vec3, MjcfError> {
        match self.floats(node, class, name)? {
            None => Ok(default),
            Some(v) if v.len() >= 3 => Ok(Vec3::new(v[0], v[1], v[2])),
            Some(_) => invalid(format!("attribute '{name}' needs 3 numbers")),
        }
    }

    fn angle(&self, a: Real) -> Real {
        if self.degrees {
            a * PI / 180.0
        } else {
            a
        }
    }

    /// Orientation from any of MuJoCo's alternative specifications.
    fn orientation(&self, node: &Node, class: &str) -> Result<Quat, MjcfError> {
        if let Some(q) = self.floats(node, class, "quat")? {
            if q.len() != 4 {
                return invalid("quat needs 4 numbers");
            }
            return Ok(Quat([q[0], q[1], q[2], q[3]]).normalized());
        }
        if let Some(a) = self.floats(node, class, "axisangle")? {
            if a.len() != 4 {
                return invalid("axisangle needs 4 numbers");
            }
            return Ok(Quat::from_axis_angle(Vec3::new(a[0], a[1], a[2]), self.angle(a[3])));
        }
        if let Some(e) = self.floats(node, class, "euler")? {
            if e.len() != 3 {
                return invalid("euler needs 3 numbers");
            }
            let mut q = Quat::IDENTITY;
            for i in 0..3 {
                let c = self.eulerseq[i];
                let axis = match c.to_ascii_lowercase() {
                    b'x' => Vec3::X,
                    b'y' => Vec3::Y,
                    b'z' => Vec3::Z,
                    _ => return invalid("bad eulerseq"),
                };
                let r = Quat::from_axis_angle(axis, self.angle(e[i]));
                q = if c.is_ascii_lowercase() { q.mul(r) } else { r.mul(q) };
            }
            return Ok(q.normalized());
        }
        if let Some(x) = self.floats(node, class, "xyaxes")? {
            if x.len() != 6 {
                return invalid("xyaxes needs 6 numbers");
            }
            let xa = Vec3::new(x[0], x[1], x[2]).normalized();
            let ya = Vec3::new(x[3], x[4], x[5]);
            let ya = (ya - xa * xa.dot(ya)).normalized();
            let za = xa.cross(ya);
            return Ok(Quat::from_mat(&Mat3::from_cols(xa, ya, za)));
        }
        if let Some(z) = self.floats(node, class, "zaxis")? {
            if z.len() != 3 {
                return invalid("zaxis needs 3 numbers");
            }
            return Ok(Quat::from_z_to(Vec3::new(z[0], z[1], z[2])));
        }
        Ok(Quat::IDENTITY)
    }
}

fn parse_floats(s: &str) -> Result<Vec<Real>, MjcfError> {
    s.split_whitespace()
        .map(|t| t.parse::<Real>().map_err(|_| MjcfError::Invalid(format!("bad number '{t}'"))))
        .collect()
}

fn children<'a, 'i>(node: Node<'a, 'i>, tag: &'a str) -> impl Iterator<Item = Node<'a, 'i>> + 'a {
    node.children().filter(move |c| c.is_element() && c.tag_name().name() == tag)
}

/// Builder state while walking the body tree.
struct Builder {
    m: Model,
    /// Per body, mass properties accumulated from geoms: (mass, m*com, second moment about body origin).
    geom_mass: Vec<(Real, Vec3, Mat3)>,
    explicit_inertial: Vec<bool>,
}

pub fn load_file(path: impl AsRef<Path>) -> Result<Model, MjcfError> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)?;
    let dir = path.parent().unwrap_or(Path::new("."));
    load_str_in(&text, dir)
}

/// Load from a string; relative asset paths resolve against the working directory.
pub fn load_str(xml: &str) -> Result<Model, MjcfError> {
    load_str_in(xml, Path::new("."))
}

/// Load from a string, resolving relative asset paths against `dir`.
pub fn load_str_in(xml: &str, dir: &Path) -> Result<Model, MjcfError> {
    let doc = Document::parse(xml)?;
    let root = doc.root_element();
    if root.tag_name().name() != "mujoco" {
        return invalid("root element must be <mujoco>");
    }

    let mut meshdir = dir.to_path_buf();
    let mut ctx = Ctx {
        degrees: true,
        eulerseq: *b"xyz",
        autolimits: true,
        defaults: HashMap::new(),
        warnings: vec![],
    };

    for c in children(root, "compiler") {
        if let Some(a) = c.attribute("angle") {
            ctx.degrees = a == "degree";
        }
        if let Some(s) = c.attribute("eulerseq") {
            let b = s.as_bytes();
            if b.len() != 3 {
                return invalid("eulerseq must have 3 characters");
            }
            ctx.eulerseq = [b[0], b[1], b[2]];
        }
        if let Some(a) = c.attribute("autolimits") {
            ctx.autolimits = a == "true";
        }
        if let Some(d) = c.attribute("meshdir").or(c.attribute("assetdir")) {
            meshdir = dir.join(d);
        }
    }

    ctx.defaults.insert("main".into(), DefaultClass::default());
    for d in children(root, "default") {
        parse_default(&mut ctx, d, "main", true);
    }

    let mut m = empty_model();
    for o in children(root, "option") {
        if let Some(v) = o.attribute("timestep") {
            m.timestep = parse_floats(v)?[0];
        }
        if let Some(v) = o.attribute("gravity") {
            let g = parse_floats(v)?;
            m.gravity = Vec3::new(g[0], g[1], g[2]);
        }
        if let Some(v) = o.attribute("solver") {
            m.solver.kind = match v {
                "Newton" => SolverKind::Newton,
                "PGS" => SolverKind::Pgs,
                other => {
                    ctx.warnings.push(format!("solver '{other}' not supported, using Newton"));
                    SolverKind::Newton
                }
            };
        }
        if let Some(v) = o.attribute("impratio") {
            m.solver.impratio = parse_floats(v)?[0];
        }
        if let Some(v) = o.attribute("cone") {
            if v != "pyramidal" {
                ctx.warnings.push(format!("cone '{v}' not supported, using pyramidal"));
            }
        }
        if let Some(v) = o.attribute("tolerance") {
            m.solver.tolerance = parse_floats(v)?[0];
        }
        if let Some(v) = o.attribute("o_solref") {
            let r = parse_floats(v)?;
            m.solver.solref = [r[0], r[1]];
        }
        if let Some(v) = o.attribute("iterations") {
            m.solver.iterations = parse_floats(v)?[0] as usize;
        }
        if let Some(v) = o.attribute("integrator") {
            if v != "Euler" {
                ctx.warnings.push(format!("integrator '{v}' not supported, using Euler"));
            }
        }
    }

    let mut b = Builder { m, geom_mass: vec![], explicit_inertial: vec![] };
    for a in children(root, "asset") {
        for mesh in children(a, "mesh") {
            parse_mesh(&ctx, &mut b.m, mesh, &meshdir)?;
        }
    }

    // world body
    add_body(&mut b, "world".into(), 0, Vec3::ZERO, Quat::IDENTITY);
    let world = match children(root, "worldbody").next() {
        Some(w) => w,
        None => return invalid("missing <worldbody>"),
    };
    parse_body_contents(&mut ctx, &mut b, world, 0, "main")?;

    finalize_inertia(&mut b)?;

    for act in children(root, "actuator") {
        for a in act.children().filter(|c| c.is_element()) {
            parse_actuator(&ctx, &mut b.m, a)?;
        }
    }
    b.m.nu = b.m.actuator_joint.len();

    for tag in ["contact", "equality", "tendon", "sensor", "keyframe", "visual", "statistic"] {
        if children(root, tag).next().is_some() {
            ctx.warnings.push(format!("<{tag}> is not supported yet and was ignored"));
        }
    }
    for w in &ctx.warnings {
        eprintln!("pippin mjcf warning: {w}");
    }

    let mut m = b.m;
    m.compile();
    crate::forward::set_const(&mut m);
    Ok(m)
}

fn parse_default(ctx: &mut Ctx, node: Node, parent: &str, is_top: bool) {
    let name = if is_top { "main".to_string() } else { node.attribute("class").unwrap_or("main").to_string() };
    // inherit from parent, then override
    let mut dc = ctx.defaults.get(parent).cloned().unwrap_or_default();
    for c in node.children().filter(|c| c.is_element() && c.tag_name().name() != "default") {
        let entry = dc.by_tag.entry(c.tag_name().name().to_string()).or_default();
        for a in c.attributes() {
            entry.insert(a.name().to_string(), a.value().to_string());
        }
    }
    ctx.defaults.insert(name.clone(), dc);
    for c in children(node, "default") {
        parse_default(ctx, c, &name, false);
    }
}

fn empty_model() -> Model {
    Model {
        timestep: 0.002,
        gravity: Vec3::new(0.0, 0.0, -9.81),
        integrator: Integrator::Euler,
        solver: SolverOptions::default(),
        nq: 0,
        nv: 0,
        nu: 0,
        body_names: vec![],
        body_parent: vec![],
        body_pos: vec![],
        body_quat: vec![],
        body_jntadr: vec![],
        body_jntnum: vec![],
        body_dofadr: vec![],
        body_dofnum: vec![],
        body_lastdof: vec![],
        body_mass: vec![],
        body_ipos: vec![],
        body_inertia: vec![],
        body_weldid: vec![],
        jnt_names: vec![],
        jnt_type: vec![],
        jnt_body: vec![],
        jnt_qposadr: vec![],
        jnt_dofadr: vec![],
        jnt_pos: vec![],
        jnt_axis: vec![],
        jnt_limited: vec![],
        jnt_range: vec![],
        jnt_stiffness: vec![],
        jnt_springref: vec![],
        dof_body: vec![],
        dof_jnt: vec![],
        dof_parent: vec![],
        dof_damping: vec![],
        dof_armature: vec![],
        geom_names: vec![],
        geom_type: vec![],
        geom_body: vec![],
        geom_pos: vec![],
        geom_quat: vec![],
        geom_size: vec![],
        geom_friction: vec![],
        geom_contype: vec![],
        geom_conaffinity: vec![],
        geom_rgba: vec![],
        geom_rbound: vec![],
        geom_dataid: vec![],
        geom_shape: vec![],
        mesh_names: vec![],
        mesh: vec![],
        mesh_hull: vec![],
        cam_names: vec![],
        cam_body: vec![],
        cam_pos: vec![],
        cam_quat: vec![],
        cam_fovy: vec![],
        actuator_names: vec![],
        actuator_joint: vec![],
        actuator_gear: vec![],
        actuator_gain: vec![],
        actuator_bias: vec![],
        actuator_ctrllimited: vec![],
        actuator_ctrlrange: vec![],
        actuator_forcelimited: vec![],
        actuator_forcerange: vec![],
        body_invweight: vec![],
        dof_invweight: vec![],
        qpos0: vec![],
        collision_pairs: vec![],
    }
}

fn add_body(b: &mut Builder, name: String, parent: usize, pos: Vec3, quat: Quat) -> usize {
    let m = &mut b.m;
    m.body_names.push(name);
    m.body_parent.push(parent);
    m.body_pos.push(pos);
    m.body_quat.push(quat);
    m.body_jntadr.push(m.jnt_type.len());
    m.body_jntnum.push(0);
    m.body_dofadr.push(m.nv);
    m.body_dofnum.push(0);
    m.body_mass.push(0.0);
    m.body_ipos.push(Vec3::ZERO);
    m.body_inertia.push(Mat3::ZERO);
    b.geom_mass.push((0.0, Vec3::ZERO, Mat3::ZERO));
    b.explicit_inertial.push(false);
    m.body_names.len() - 1
}

fn parse_body_contents(ctx: &mut Ctx, b: &mut Builder, node: Node, body: usize, class: &str) -> Result<(), MjcfError> {
    for c in node.children().filter(|c| c.is_element()) {
        match c.tag_name().name() {
            "geom" => parse_geom(ctx, b, c, body, class)?,
            "joint" | "freejoint" => {
                if body == 0 {
                    return invalid("joints cannot be attached to the world body");
                }
                parse_joint(ctx, b, c, body, class)?
            }
            "inertial" => parse_inertial(ctx, b, c, body, class)?,
            "body" => {
                let name = c.attribute("name").map(String::from).unwrap_or_else(|| format!("body{}", b.m.nbody()));
                let pos = ctx.vec3(&c, class, "pos", Vec3::ZERO)?;
                let quat = ctx.orientation(&c, class)?;
                let child_class = c.attribute("childclass").unwrap_or(class).to_string();
                let id = add_body(b, name, body, pos, quat);
                parse_body_contents(ctx, b, c, id, &child_class)?;
            }
            "camera" => {
                let m = &mut b.m;
                let id = m.cam_body.len();
                m.cam_names.push(c.attribute("name").map(String::from).unwrap_or_else(|| format!("camera{id}")));
                m.cam_body.push(body);
                m.cam_pos.push(ctx.vec3(&c, class, "pos", Vec3::ZERO)?);
                m.cam_quat.push(ctx.orientation(&c, class)?);
                // fovy is always in degrees in MJCF
                m.cam_fovy.push(ctx.float(&c, class, "fovy", 45.0)?);
            }
            "site" | "light" | "include" => {}
            other => ctx.warnings.push(format!("<{other}> inside body is not supported")),
        }
    }
    Ok(())
}

fn parse_joint(ctx: &mut Ctx, b: &mut Builder, node: Node, body: usize, class: &str) -> Result<(), MjcfError> {
    let jt = if node.tag_name().name() == "freejoint" {
        JointType::Free
    } else {
        match ctx.get(&node, class, "type").unwrap_or("hinge") {
            "free" => JointType::Free,
            "ball" => JointType::Ball,
            "slide" => JointType::Slide,
            "hinge" => JointType::Hinge,
            t => return invalid(format!("unknown joint type '{t}'")),
        }
    };
    let m = &mut b.m;
    let jid = m.jnt_type.len();
    let name = node.attribute("name").map(String::from).unwrap_or_else(|| format!("joint{jid}"));
    let pos = if jt == JointType::Free { Vec3::ZERO } else { ctx.vec3(&node, class, "pos", Vec3::ZERO)? };
    let axis = ctx.vec3(&node, class, "axis", Vec3::Z)?.normalized();
    let range = ctx.floats(&node, class, "range")?;
    let limited_attr = ctx.get(&node, class, "limited");
    let limited = match limited_attr {
        Some("true") => true,
        Some("false") => false,
        _ => ctx.autolimits && range.is_some(),
    } && matches!(jt, JointType::Hinge | JointType::Slide);
    let mut range = range.map(|r| [r[0], r[1]]).unwrap_or([0.0, 0.0]);
    let is_angular = jt == JointType::Hinge;
    if is_angular {
        range = [ctx.angle(range[0]), ctx.angle(range[1])];
    }
    let mut refv = ctx.float(&node, class, "ref", 0.0)?;
    let mut springref = ctx.float(&node, class, "springref", 0.0)?;
    if is_angular {
        refv = ctx.angle(refv);
        springref = ctx.angle(springref);
    }
    let damping = ctx.float(&node, class, "damping", 0.0)?;
    let armature = ctx.float(&node, class, "armature", 0.0)?;
    let stiffness = ctx.float(&node, class, "stiffness", 0.0)?;

    m.jnt_names.push(name);
    m.jnt_type.push(jt);
    m.jnt_body.push(body);
    m.jnt_qposadr.push(m.nq);
    m.jnt_dofadr.push(m.nv);
    m.jnt_pos.push(pos);
    m.jnt_axis.push(axis);
    m.jnt_limited.push(limited);
    m.jnt_range.push(range);
    m.jnt_stiffness.push(stiffness);
    m.jnt_springref.push(springref);
    match jt {
        JointType::Free => {
            let p = m.body_pos[body];
            let q = m.body_quat[body];
            m.qpos0.extend_from_slice(&[p[0], p[1], p[2], q.0[0], q.0[1], q.0[2], q.0[3]]);
        }
        JointType::Ball => m.qpos0.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]),
        _ => m.qpos0.push(refv),
    }
    for _ in 0..jt.nv() {
        m.dof_body.push(body);
        m.dof_jnt.push(jid);
        m.dof_damping.push(damping);
        m.dof_armature.push(armature);
    }
    m.nq += jt.nq();
    m.nv += jt.nv();
    m.body_jntnum[body] += 1;
    m.body_dofnum[body] += jt.nv();
    if m.body_jntnum[body] == 1 {
        m.body_jntadr[body] = jid;
        m.body_dofadr[body] = m.jnt_dofadr[jid];
    }
    Ok(())
}

fn parse_geom(ctx: &mut Ctx, b: &mut Builder, node: Node, body: usize, class: &str) -> Result<(), MjcfError> {
    let gt = match ctx.get(&node, class, "type").unwrap_or("sphere") {
        "plane" => GeomType::Plane,
        "sphere" => GeomType::Sphere,
        "capsule" => GeomType::Capsule,
        "box" => GeomType::Box,
        "cylinder" => GeomType::Cylinder,
        "mesh" => GeomType::Mesh,
        t => {
            ctx.warnings.push(format!("geom type '{t}' not supported yet; geom skipped"));
            return Ok(());
        }
    };
    let sz = ctx.floats(&node, class, "size")?.unwrap_or_default();
    let get = |i: usize| sz.get(i).copied().unwrap_or(0.0);
    let mut size = Vec3::new(get(0), get(1), get(2));
    let mut pos = ctx.vec3(&node, class, "pos", Vec3::ZERO)?;
    let mut quat = ctx.orientation(&node, class)?;

    if let Some(ft) = ctx.floats(&node, class, "fromto")? {
        if ft.len() != 6 {
            return invalid("fromto needs 6 numbers");
        }
        let a = Vec3::new(ft[0], ft[1], ft[2]);
        let bb = Vec3::new(ft[3], ft[4], ft[5]);
        let d = bb - a;
        pos = (a + bb) * 0.5;
        quat = Quat::from_z_to(d);
        match gt {
            GeomType::Capsule | GeomType::Cylinder => size = Vec3::new(get(0), d.norm() * 0.5, 0.0),
            GeomType::Box => size = Vec3::new(get(0), get(1), d.norm() * 0.5),
            _ => return invalid("fromto only valid for capsule, cylinder, box"),
        }
    }

    let friction = ctx.floats(&node, class, "friction")?.and_then(|f| f.first().copied()).unwrap_or(1.0);
    let contype = ctx.float(&node, class, "contype", 1.0)? as u32;
    let conaffinity = ctx.float(&node, class, "conaffinity", 1.0)? as u32;
    let rgba = ctx.floats(&node, class, "rgba")?.map(|v| [v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32]);

    let dataid = if gt == GeomType::Mesh {
        let name = ctx.get(&node, class, "mesh").ok_or_else(|| MjcfError::Invalid("mesh geom needs a 'mesh' attribute".into()))?;
        b.m.mesh_names.iter().position(|n| n == name).ok_or_else(|| MjcfError::Invalid(format!("unknown mesh '{name}'")))?
    } else {
        usize::MAX
    };

    // mass properties
    if gt == GeomType::Mesh {
        let (vol, com, icom) = crate::mesh::mass_properties(&b.m.mesh_hull[dataid]);
        let mass = match ctx.floats(&node, class, "mass")? {
            Some(v) => v[0],
            None => ctx.float(&node, class, "density", 1000.0)? * vol,
        };
        if mass > 0.0 && vol > 0.0 {
            let r = quat.to_mat();
            let c = pos + r.mul_vec(com);
            let ig = icom.scale(mass / vol).rotate(&r);
            let cc = Mat3::skew(c);
            let shifted = ig.add(&cc.mul_mat(&cc.transpose()).scale(mass));
            let e = &mut b.geom_mass[body];
            e.0 += mass;
            e.1 += c * mass;
            e.2 = e.2.add(&shifted);
        }
    } else if gt != GeomType::Plane {
        let (vol, inertia_unit) = geom_volume_inertia(gt, size);
        let mass = match ctx.floats(&node, class, "mass")? {
            Some(v) => v[0],
            None => ctx.float(&node, class, "density", 1000.0)? * vol,
        };
        if mass > 0.0 && vol > 0.0 {
            // inertia about geom center in geom frame, scaled to mass
            let ig = Mat3::diag(inertia_unit * (mass / vol)).rotate(&quat.to_mat());
            // second moment about body origin: I_c + m (|p|^2 I - p p^T)
            let pp = Mat3::skew(pos);
            let shifted = ig.add(&pp.mul_mat(&pp.transpose()).scale(mass));
            let e = &mut b.geom_mass[body];
            e.0 += mass;
            e.1 += pos * mass;
            e.2 = e.2.add(&shifted);
        }
    }

    let m = &mut b.m;
    let gid = m.geom_type.len();
    m.geom_names.push(node.attribute("name").map(String::from).unwrap_or_else(|| format!("geom{gid}")));
    m.geom_type.push(gt);
    m.geom_body.push(body);
    m.geom_pos.push(pos);
    m.geom_quat.push(quat);
    m.geom_size.push(size);
    m.geom_friction.push(friction);
    m.geom_contype.push(contype);
    m.geom_conaffinity.push(conaffinity);
    m.geom_rgba.push(rgba.unwrap_or([0.5, 0.5, 0.5, 1.0]));
    m.geom_dataid.push(dataid);
    Ok(())
}

fn parse_mesh(ctx: &Ctx, m: &mut Model, node: Node, meshdir: &Path) -> Result<(), MjcfError> {
    let class = "main";
    let file = node.attribute("file");
    let name = node
        .attribute("name")
        .map(String::from)
        .or_else(|| file.and_then(|f| Path::new(f).file_stem()).map(|s| s.to_string_lossy().into_owned()))
        .ok_or_else(|| MjcfError::Invalid("mesh needs a name or file".into()))?;
    let mut mesh = if let Some(f) = file {
        crate::mesh::load(&meshdir.join(f)).map_err(|e| MjcfError::Invalid(e.to_string()))?
    } else if let Some(v) = ctx.floats(&node, class, "vertex")? {
        let vertices: Vec<Vec3> = v.chunks(3).filter(|c| c.len() == 3).map(|c| Vec3::new(c[0], c[1], c[2])).collect();
        crate::mesh::TriMesh { vertices, triangles: vec![] }
    } else {
        return invalid(format!("mesh '{name}' needs 'file' or 'vertex'"));
    };
    let scale = ctx.vec3(&node, class, "scale", Vec3::new(1.0, 1.0, 1.0))?;
    for v in &mut mesh.vertices {
        *v = v.mul_elem(scale);
    }
    let hull = crate::mesh::convex_hull(&mesh.vertices);
    if hull.triangles.is_empty() {
        return invalid(crate::mesh::MeshError::Degenerate(name).to_string());
    }
    if mesh.triangles.is_empty() {
        mesh = hull.clone(); // point cloud: render the hull
    }
    m.mesh_names.push(name);
    m.mesh.push(mesh);
    m.mesh_hull.push(hull);
    Ok(())
}

/// Volume and principal inertia per unit density (i.e. inertia / density).
fn geom_volume_inertia(gt: GeomType, s: Vec3) -> (Real, Vec3) {
    match gt {
        GeomType::Plane | GeomType::Mesh => (0.0, Vec3::ZERO),
        GeomType::Sphere => {
            let r = s[0];
            let v = 4.0 / 3.0 * PI * r * r * r;
            let i = 0.4 * v * r * r;
            (v, Vec3::new(i, i, i))
        }
        GeomType::Box => {
            let (a, b, c) = (s[0], s[1], s[2]);
            let v = 8.0 * a * b * c;
            (v, Vec3::new(v * (b * b + c * c) / 3.0, v * (a * a + c * c) / 3.0, v * (a * a + b * b) / 3.0))
        }
        GeomType::Cylinder => {
            let (r, h) = (s[0], s[1]);
            let v = PI * r * r * 2.0 * h;
            let ix = v * (3.0 * r * r + 4.0 * h * h) / 12.0;
            (v, Vec3::new(ix, ix, 0.5 * v * r * r))
        }
        GeomType::Capsule => {
            let (r, h) = (s[0], s[1]);
            let vc = PI * r * r * 2.0 * h;
            let vs = 4.0 / 3.0 * PI * r * r * r;
            let v = vc + vs;
            // cylinder + two hemispheres (exact)
            let hs = vs * 0.5;
            let d = h + 3.0 * r / 8.0;
            let hs_ix = hs * (83.0 / 320.0) * r * r;
            let ix = vc * (3.0 * r * r + 4.0 * h * h) / 12.0 + 2.0 * (hs_ix + hs * d * d);
            let iz = 0.5 * vc * r * r + 0.4 * vs * r * r;
            (v, Vec3::new(ix, ix, iz))
        }
    }
}

fn parse_inertial(ctx: &mut Ctx, b: &mut Builder, node: Node, body: usize, class: &str) -> Result<(), MjcfError> {
    let pos = ctx.vec3(&node, class, "pos", Vec3::ZERO)?;
    let quat = ctx.orientation(&node, class)?;
    let mass = ctx.float(&node, class, "mass", 0.0)?;
    let inertia = if let Some(d) = ctx.floats(&node, class, "diaginertia")? {
        Mat3::diag(Vec3::new(d[0], d[1], d[2])).rotate(&quat.to_mat())
    } else if let Some(f) = ctx.floats(&node, class, "fullinertia")? {
        // ixx iyy izz ixy ixz iyz
        Mat3([f[0], f[3], f[4], f[3], f[1], f[5], f[4], f[5], f[2]]).rotate(&quat.to_mat())
    } else {
        return invalid("<inertial> requires diaginertia or fullinertia");
    };
    b.m.body_mass[body] = mass;
    b.m.body_ipos[body] = pos;
    b.m.body_inertia[body] = inertia;
    b.explicit_inertial[body] = true;
    Ok(())
}

fn finalize_inertia(b: &mut Builder) -> Result<(), MjcfError> {
    for body in 1..b.m.nbody() {
        if b.explicit_inertial[body] {
            continue;
        }
        let (mass, mc, second) = b.geom_mass[body];
        if mass <= 0.0 {
            if b.m.body_jntnum[body] > 0 {
                return invalid(format!("moving body '{}' has no mass", b.m.body_names[body]));
            }
            continue;
        }
        let com = mc * (1.0 / mass);
        let cx = Mat3::skew(com);
        // parallel-axis back to com
        let icom = second.add(&cx.mul_mat(&cx.transpose()).scale(-mass));
        b.m.body_mass[body] = mass;
        b.m.body_ipos[body] = com;
        b.m.body_inertia[body] = icom;
    }
    Ok(())
}

fn parse_actuator(ctx: &Ctx, m: &mut Model, node: Node) -> Result<(), MjcfError> {
    let tag = node.tag_name().name();
    if !matches!(tag, "motor" | "position" | "velocity" | "general") {
        return invalid(format!("actuator <{tag}> not supported"));
    }
    let class = "main";
    let Some(jname) = node.attribute("joint") else {
        return invalid("only joint transmissions are supported");
    };
    let Some(j) = m.joint_id(jname) else {
        return invalid(format!("actuator references unknown joint '{jname}'"));
    };
    if !matches!(m.jnt_type[j], JointType::Hinge | JointType::Slide) {
        return invalid("actuators on free/ball joints are not supported");
    }
    let gear = ctx.float(&node, class, "gear", 1.0)?;
    let (gain, bias) = match tag {
        "motor" => (1.0, [0.0, 0.0, 0.0]),
        "position" => {
            let kp = ctx.float(&node, class, "kp", 1.0)?;
            let kv = ctx.float(&node, class, "kv", 0.0)?;
            (kp, [0.0, -kp, -kv])
        }
        "velocity" => {
            let kv = ctx.float(&node, class, "kv", 1.0)?;
            (kv, [0.0, 0.0, -kv])
        }
        _ => {
            let g = ctx.floats(&node, class, "gainprm")?.map(|v| v[0]).unwrap_or(1.0);
            let bp = ctx.floats(&node, class, "biasprm")?.unwrap_or_default();
            let bias = if ctx.get(&node, class, "biastype") == Some("affine") {
                [bp.first().copied().unwrap_or(0.0), bp.get(1).copied().unwrap_or(0.0), bp.get(2).copied().unwrap_or(0.0)]
            } else {
                [0.0; 3]
            };
            (g, bias)
        }
    };
    let limited = |flag: &str, range: &str| -> Result<(bool, [Real; 2]), MjcfError> {
        let r = ctx.floats(&node, class, range)?;
        let lim = match ctx.get(&node, class, flag) {
            Some("true") => true,
            Some("false") => false,
            _ => ctx.autolimits && r.is_some(),
        };
        Ok((lim, r.map(|r| [r[0], r[1]]).unwrap_or([0.0, 0.0])))
    };
    let (cl, cr) = limited("ctrllimited", "ctrlrange")?;
    let (fl, fr) = limited("forcelimited", "forcerange")?;
    let idx = m.actuator_joint.len();
    m.actuator_names.push(node.attribute("name").map(String::from).unwrap_or_else(|| format!("actuator{idx}")));
    m.actuator_joint.push(j);
    m.actuator_gear.push(gear);
    m.actuator_gain.push(gain);
    m.actuator_bias.push(bias);
    m.actuator_ctrllimited.push(cl);
    m.actuator_ctrlrange.push(cr);
    m.actuator_forcelimited.push(fl);
    m.actuator_forcerange.push(fr);
    Ok(())
}
