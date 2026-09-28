//! URDF import, by conversion to MJCF (the approach MuJoCo itself takes).
//!
//! - links become bodies; joints become hinge (revolute, continuous), slide
//!   (prismatic), or no joint (fixed). The root link is welded to the world
//!   unless `floating_base` is set.
//! - `<collision>` geometry collides; `<visual>` geometry is render-only
//!   (contype/conaffinity 0, massless, group 1).
//! - `<inertial>` is used as given; links without one get mass from their
//!   collision geometry.
//! - mesh filenames may use `package://`, `file://`, or relative paths.

use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::path::{Path, PathBuf};

use roxmltree::{Document, Node};

use crate::math::{Quat, Vec3};
use crate::mjcf::MjcfError;

#[derive(Clone, Debug, Default)]
pub struct UrdfOptions {
    /// Give the root link a free joint (mobile robots, loose objects).
    pub floating_base: bool,
    /// Directories searched for `package://<pkg>/...` (each may contain
    /// `<pkg>/` or be the package itself). `ROS_PACKAGE_PATH` is also used.
    pub package_dirs: Vec<PathBuf>,
}

fn invalid<T>(msg: impl Into<String>) -> Result<T, MjcfError> {
    Err(MjcfError::Invalid(msg.into()))
}

fn floats(s: &str) -> Result<Vec<f64>, MjcfError> {
    s.split_whitespace().map(|t| t.parse::<f64>().map_err(|_| MjcfError::Invalid(format!("bad number '{t}' in URDF")))).collect()
}

fn attr3(node: Option<Node>, name: &str, default: [f64; 3]) -> Result<[f64; 3], MjcfError> {
    match node.and_then(|n| n.attribute(name)) {
        None => Ok(default),
        Some(s) => {
            let v = floats(s)?;
            if v.len() != 3 {
                return invalid(format!("'{name}' needs 3 numbers"));
            }
            Ok([v[0], v[1], v[2]])
        }
    }
}

fn child<'a, 'i>(n: Node<'a, 'i>, tag: &str) -> Option<Node<'a, 'i>> {
    n.children().find(|c| c.is_element() && c.tag_name().name() == tag)
}

fn children<'a, 'i>(n: Node<'a, 'i>, tag: &'static str) -> impl Iterator<Item = Node<'a, 'i>> {
    n.children().filter(move |c| c.is_element() && c.tag_name().name() == tag)
}

/// URDF rpy (fixed-axis roll, pitch, yaw) as a quaternion: R = Rz Ry Rx.
fn rpy_quat(rpy: [f64; 3]) -> Quat {
    let q = |axis: Vec3, a: f64| Quat::from_axis_angle(axis, a);
    q(Vec3::Z, rpy[2]).mul(q(Vec3::Y, rpy[1])).mul(q(Vec3::X, rpy[0])).normalized()
}

/// ` pos="x y z" quat="w x y z"` for an `<origin>` element.
fn origin_attrs(origin: Option<Node>) -> Result<String, MjcfError> {
    let xyz = attr3(origin, "xyz", [0.0; 3])?;
    let q = rpy_quat(attr3(origin, "rpy", [0.0; 3])?);
    Ok(format!(
        r#" pos="{} {} {}" quat="{} {} {} {}""#,
        xyz[0], xyz[1], xyz[2], q.0[0], q.0[1], q.0[2], q.0[3]
    ))
}

struct Converter<'a> {
    base: PathBuf,
    opts: &'a UrdfOptions,
    materials: HashMap<String, [f64; 4]>,
    meshes: Vec<(String, PathBuf, [f64; 3])>,
    mesh_ids: HashMap<(PathBuf, [u64; 3]), String>,
    warnings: Vec<String>,
}

impl Converter<'_> {
    fn resolve(&self, filename: &str) -> Option<PathBuf> {
        if let Some(rest) = filename.strip_prefix("package://") {
            let (pkg, rel) = rest.split_once('/')?;
            let mut dirs: Vec<PathBuf> = self.opts.package_dirs.clone();
            if let Ok(p) = std::env::var("ROS_PACKAGE_PATH") {
                dirs.extend(p.split(':').map(PathBuf::from));
            }
            // conventional layouts relative to the URDF file
            for up in [0, 1, 2, 3] {
                let mut d = self.base.clone();
                for _ in 0..up {
                    d.pop();
                }
                dirs.push(d);
            }
            for d in dirs {
                for cand in [d.join(pkg).join(rel), d.join(rel)] {
                    if cand.exists() {
                        return std::fs::canonicalize(&cand).ok();
                    }
                }
            }
            None
        } else {
            let p = filename.strip_prefix("file://").unwrap_or(filename);
            let p = if Path::new(p).is_absolute() { PathBuf::from(p) } else { self.base.join(p) };
            // absolute, so the generated MJCF is independent of where it is loaded from
            std::fs::canonicalize(&p).ok()
        }
    }

    fn mesh(&mut self, filename: &str, scale: [f64; 3]) -> Option<String> {
        let path = match self.resolve(filename) {
            Some(p) => p,
            None => {
                self.warnings.push(format!("mesh '{filename}' not found; geom skipped"));
                return None;
            }
        };
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
        if ext != "stl" && ext != "obj" {
            self.warnings.push(format!("mesh '{filename}': .{ext} not supported (use STL/OBJ); geom skipped"));
            return None;
        }
        let key = (path.clone(), scale.map(f64::to_bits));
        if let Some(name) = self.mesh_ids.get(&key) {
            return Some(name.clone());
        }
        let name = format!("urdf_mesh{}", self.meshes.len());
        self.meshes.push((name.clone(), path, scale));
        self.mesh_ids.insert(key, name.clone());
        Some(name)
    }

    /// `<geom .../>` for a visual or collision element, or None if skipped.
    /// `hide` puts collision geometry in group 3 (hidden) when the link has visuals.
    fn geom(&mut self, el: Node, visual: bool, hide: bool) -> Result<Option<String>, MjcfError> {
        let Some(geometry) = child(el, "geometry") else { return Ok(None) };
        let Some(shape) = geometry.children().find(|c| c.is_element()) else { return Ok(None) };
        let mut g = String::from("<geom");
        match shape.tag_name().name() {
            "box" => {
                let s = attr3(Some(shape), "size", [0.0; 3])?;
                write!(g, r#" type="box" size="{} {} {}""#, s[0] / 2.0, s[1] / 2.0, s[2] / 2.0).unwrap();
            }
            "cylinder" => {
                let r = floats(shape.attribute("radius").unwrap_or("0"))?[0];
                let l = floats(shape.attribute("length").unwrap_or("0"))?[0];
                write!(g, r#" type="cylinder" size="{r} {}""#, l / 2.0).unwrap();
            }
            "sphere" => {
                let r = floats(shape.attribute("radius").unwrap_or("0"))?[0];
                write!(g, r#" type="sphere" size="{r}""#).unwrap();
            }
            "capsule" => {
                let r = floats(shape.attribute("radius").unwrap_or("0"))?[0];
                let l = floats(shape.attribute("length").unwrap_or("0"))?[0];
                write!(g, r#" type="capsule" size="{r} {}""#, l / 2.0).unwrap();
            }
            "mesh" => {
                let file = shape.attribute("filename").unwrap_or("");
                let scale = attr3(Some(shape), "scale", [1.0; 3])?;
                match self.mesh(file, scale) {
                    Some(name) => write!(g, r#" type="mesh" mesh="{name}""#).unwrap(),
                    None => return Ok(None),
                }
            }
            other => {
                self.warnings.push(format!("URDF geometry <{other}> not supported; skipped"));
                return Ok(None);
            }
        }
        g += &origin_attrs(child(el, "origin"))?;
        if visual {
            g += r#" contype="0" conaffinity="0" group="1" mass="0""#;
            if let Some(rgba) = self.material_rgba(el) {
                write!(g, r#" rgba="{} {} {} {}""#, rgba[0], rgba[1], rgba[2], rgba[3]).unwrap();
            }
        } else if hide {
            g += r#" group="3""#;
        }
        g += "/>";
        Ok(Some(g))
    }

    fn material_rgba(&self, visual: Node) -> Option<[f64; 4]> {
        let mat = child(visual, "material")?;
        if let Some(c) = child(mat, "color").and_then(|c| c.attribute("rgba")) {
            let v = floats(c).ok()?;
            return (v.len() == 4).then(|| [v[0], v[1], v[2], v[3]]);
        }
        self.materials.get(mat.attribute("name")?).copied()
    }
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('"', "&quot;").replace('<', "&lt;").replace('>', "&gt;")
}

/// Convert URDF XML to an MJCF string. `base` is the URDF file's directory.
pub fn to_mjcf(urdf: &str, base: &Path, opts: &UrdfOptions) -> Result<(String, Vec<String>), MjcfError> {
    let doc = Document::parse(urdf)?;
    let robot = doc.root_element();
    if robot.tag_name().name() != "robot" {
        return invalid("URDF root element must be <robot>");
    }
    let mut cv = Converter {
        base: base.to_path_buf(),
        opts,
        materials: HashMap::new(),
        meshes: vec![],
        mesh_ids: HashMap::new(),
        warnings: vec![],
    };
    for m in children(robot, "material") {
        if let (Some(name), Some(c)) = (m.attribute("name"), child(m, "color").and_then(|c| c.attribute("rgba"))) {
            let v = floats(c)?;
            if v.len() == 4 {
                cv.materials.insert(name.to_string(), [v[0], v[1], v[2], v[3]]);
            }
        }
    }

    let links: HashMap<&str, Node> = children(robot, "link").filter_map(|l| Some((l.attribute("name")?, l))).collect();
    let link_order: Vec<&str> = children(robot, "link").filter_map(|l| l.attribute("name")).collect();
    let mut kids: HashMap<&str, Vec<Node>> = HashMap::new();
    let mut is_child: HashSet<&str> = HashSet::new();
    for j in children(robot, "joint") {
        let parent = child(j, "parent").and_then(|p| p.attribute("link"));
        let ch = child(j, "child").and_then(|c| c.attribute("link"));
        let (Some(p), Some(c)) = (parent, ch) else { return invalid("joint missing parent/child") };
        if !links.contains_key(p) || !links.contains_key(c) {
            return invalid(format!("joint '{}' references unknown link", j.attribute("name").unwrap_or("?")));
        }
        if !is_child.insert(c) {
            return invalid(format!("link '{c}' has two parent joints (URDF must be a tree)"));
        }
        kids.entry(p).or_default().push(j);
    }
    let roots: Vec<&str> = link_order.iter().copied().filter(|l| !is_child.contains(l)).collect();
    if roots.len() != 1 {
        return invalid(format!("URDF must have exactly one root link, found {roots:?}"));
    }

    let mut body = String::new();
    fn emit_link(cv: &mut Converter, links: &HashMap<&str, Node>, kids: &HashMap<&str, Vec<Node>>, link: &str, out: &mut String) -> Result<(), MjcfError> {
        let l = links[link];
        if let Some(inertial) = child(l, "inertial") {
            let mass = child(inertial, "mass").and_then(|m| m.attribute("value")).map(floats).transpose()?.map(|v| v[0]).unwrap_or(0.0);
            let i = child(inertial, "inertia");
            let get = |k: &str| i.and_then(|i| i.attribute(k)).map(floats).transpose().map(|v| v.map(|v| v[0]).unwrap_or(0.0));
            let (ixx, iyy, izz, ixy, ixz, iyz) = (get("ixx")?, get("iyy")?, get("izz")?, get("ixy")?, get("ixz")?, get("iyz")?);
            if mass > 0.0 {
                write!(
                    out,
                    r#"<inertial{} mass="{mass}" fullinertia="{ixx} {iyy} {izz} {ixy} {ixz} {iyz}"/>"#,
                    origin_attrs(child(inertial, "origin"))?
                )
                .unwrap();
            }
        }
        let has_visual = children(l, "visual").next().is_some();
        for c in children(l, "collision") {
            if let Some(g) = cv.geom(c, false, has_visual)? {
                out.push_str(&g);
            }
        }
        for v in children(l, "visual") {
            if let Some(g) = cv.geom(v, true, false)? {
                out.push_str(&g);
            }
        }
        for j in kids.get(link).map(Vec::as_slice).unwrap_or(&[]) {
            let name = j.attribute("name").unwrap_or("joint");
            let ch = child(*j, "child").and_then(|c| c.attribute("link")).unwrap();
            write!(out, r#"<body name="{}"{}>"#, escape(ch), origin_attrs(child(*j, "origin"))?).unwrap();
            let axis = attr3(child(*j, "axis"), "xyz", [1.0, 0.0, 0.0])?;
            let damping = child(*j, "dynamics").and_then(|d| d.attribute("damping")).map(floats).transpose()?.map(|v| v[0]).unwrap_or(0.0);
            let limit = child(*j, "limit");
            let range = |lim: Option<Node>| -> Result<Option<(f64, f64)>, MjcfError> {
                let lo = lim.and_then(|l| l.attribute("lower")).map(floats).transpose()?.map(|v| v[0]).unwrap_or(0.0);
                let hi = lim.and_then(|l| l.attribute("upper")).map(floats).transpose()?.map(|v| v[0]).unwrap_or(0.0);
                Ok((lo < hi).then_some((lo, hi)))
            };
            let name = escape(name);
            let common = format!(r#"name="{name}" axis="{} {} {}" damping="{damping}""#, axis[0], axis[1], axis[2]);
            match j.attribute("type").unwrap_or("fixed") {
                "revolute" => match range(limit)? {
                    Some((lo, hi)) => write!(out, r#"<joint type="hinge" {common} range="{lo} {hi}"/>"#).unwrap(),
                    None => write!(out, r#"<joint type="hinge" {common}/>"#).unwrap(),
                },
                "continuous" => write!(out, r#"<joint type="hinge" {common}/>"#).unwrap(),
                "prismatic" => match range(limit)? {
                    Some((lo, hi)) => write!(out, r#"<joint type="slide" {common} range="{lo} {hi}"/>"#).unwrap(),
                    None => write!(out, r#"<joint type="slide" {common}/>"#).unwrap(),
                },
                "fixed" => {}
                "floating" => write!(out, r#"<freejoint name="{name}"/>"#).unwrap(),
                other => cv.warnings.push(format!("joint type '{other}' not supported; treated as fixed")),
            }
            emit_link(cv, links, kids, ch, out)?;
            out.push_str("</body>");
        }
        Ok(())
    }
    let root = roots[0];
    write!(body, r#"<body name="{}">"#, escape(root)).unwrap();
    if opts.floating_base {
        body.push_str(r#"<freejoint name="root"/>"#);
    }
    emit_link(&mut cv, &links, &kids, root, &mut body)?;
    body.push_str("</body>");

    let mut assets = String::new();
    for (name, path, s) in &cv.meshes {
        write!(assets, r#"<mesh name="{name}" file="{}" scale="{} {} {}"/>"#, escape(&path.to_string_lossy()), s[0], s[1], s[2]).unwrap();
    }
    let name = robot.attribute("name").unwrap_or("robot");
    let mjcf = format!(
        r#"<mujoco model="{name}"><compiler angle="radian" autolimits="true"/><asset>{assets}</asset><worldbody>{body}</worldbody></mujoco>"#
    );
    Ok((mjcf, cv.warnings))
}

/// Load a URDF file.
pub fn load_file(path: impl AsRef<Path>, opts: &UrdfOptions) -> Result<crate::Model, MjcfError> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)?;
    let base = path.parent().unwrap_or(Path::new("."));
    let (mjcf, warnings) = to_mjcf(&text, base, opts)?;
    for w in warnings {
        eprintln!("pippin urdf warning: {w}");
    }
    crate::mjcf::load_str_in(&mjcf, base)
}
