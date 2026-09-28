//! Triangle meshes: loading (STL, OBJ, inline vertices), convex hulls, and
//! exact solid mass properties.
//!
//! As in MuJoCo, meshes collide through their convex hull. Unlike MuJoCo, a
//! mesh keeps its file coordinates: the geom frame is the mesh frame (MuJoCo
//! re-centers meshes on their center of mass). Body dynamics are identical.

use std::path::Path;

use parry3d_f64::math::Vector;

use crate::math::{Mat3, Real, Vec3};

#[derive(Clone, Debug, Default)]
pub struct TriMesh {
    pub vertices: Vec<Vec3>,
    pub triangles: Vec<[u32; 3]>,
}

#[derive(Debug, thiserror::Error)]
pub enum MeshError {
    #[error("cannot read mesh '{0}': {1}")]
    Io(String, String),
    #[error("unsupported mesh format '{0}' (use .stl or .obj)")]
    Format(String),
    #[error("mesh '{0}' is degenerate (no volume)")]
    Degenerate(String),
}

pub fn load(path: &Path) -> Result<TriMesh, MeshError> {
    let name = path.display().to_string();
    let io = |e: &dyn std::fmt::Display| MeshError::Io(name.clone(), e.to_string());
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        "stl" => {
            let mut f = std::fs::File::open(path).map_err(|e| io(&e))?;
            let m = stl_io::read_stl(&mut f).map_err(|e| io(&e))?;
            Ok(TriMesh {
                vertices: m.vertices.iter().map(|v| Vec3::new(v[0] as Real, v[1] as Real, v[2] as Real)).collect(),
                triangles: m.faces.iter().map(|t| t.vertices.map(|i| i as u32)).collect(),
            })
        }
        "obj" => {
            let opts = tobj::LoadOptions { triangulate: true, single_index: true, ..Default::default() };
            let (models, _) = tobj::load_obj(path, &opts).map_err(|e| io(&e))?;
            let mut out = TriMesh::default();
            for m in models {
                let base = out.vertices.len() as u32;
                out.vertices.extend(m.mesh.positions.chunks(3).map(|p| Vec3::new(p[0] as Real, p[1] as Real, p[2] as Real)));
                out.triangles.extend(m.mesh.indices.chunks(3).map(|t| [base + t[0], base + t[1], base + t[2]]));
            }
            Ok(out)
        }
        other => Err(MeshError::Format(other.to_string())),
    }
}

fn to_parry(v: Vec3) -> Vector {
    Vector::new(v[0], v[1], v[2])
}

/// Convex hull of a point set, as a triangle mesh with outward winding.
pub fn convex_hull(points: &[Vec3]) -> TriMesh {
    let pts: Vec<Vector> = points.iter().map(|&p| to_parry(p)).collect();
    let (v, t) = parry3d_f64::transformation::convex_hull(&pts);
    TriMesh { vertices: v.iter().map(|p| Vec3::new(p.x, p.y, p.z)).collect(), triangles: t }
}

/// Volume, center of mass, and inertia about the center of mass (per unit
/// density) of a closed, outward-wound triangle mesh, by signed tetrahedra.
pub fn mass_properties(m: &TriMesh) -> (Real, Vec3, Mat3) {
    let mut vol = 0.0;
    let mut first = Vec3::ZERO;
    // second moments about the origin: integral of x x^T dV
    let mut c = [[0.0 as Real; 3]; 3];
    for t in &m.triangles {
        let [a, b, d] = t.map(|i| m.vertices[i as usize]);
        let v6 = a.dot(b.cross(d)); // 6 x signed volume of (0, a, b, d)
        vol += v6 / 6.0;
        first += (a + b + d) * (v6 / 24.0);
        for i in 0..3 {
            for j in 0..3 {
                // integral over tetrahedron (0,a,b,d) of x_i x_j
                let s = a[i] * a[j] + b[i] * b[j] + d[i] * d[j];
                let p = (a[i] + b[i] + d[i]) * (a[j] + b[j] + d[j]);
                c[i][j] += v6 / 120.0 * (s + p);
            }
        }
    }
    if vol.abs() < 1e-18 {
        return (0.0, Vec3::ZERO, Mat3::ZERO);
    }
    let com = first * (1.0 / vol);
    // inertia about origin: I = tr(C) 1 - C, then parallel-axis to com
    let tr = c[0][0] + c[1][1] + c[2][2];
    let mut io = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            io[3 * i + j] = if i == j { tr } else { 0.0 } - c[i][j];
        }
    }
    let cx = Mat3::skew(com);
    let icom = Mat3(io).add(&cx.mul_mat(&cx.transpose()).scale(-vol));
    (vol, com, icom)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube(h: Real) -> TriMesh {
        let pts: Vec<Vec3> =
            (0..8).map(|i| Vec3::new(if i & 1 == 0 { -h } else { h }, if i & 2 == 0 { -h } else { h }, if i & 4 == 0 { -h } else { h })).collect();
        convex_hull(&pts)
    }

    #[test]
    fn cube_mass_properties_are_exact() {
        let h = 0.3;
        let offset = Vec3::new(1.0, -2.0, 0.5);
        let mut m = cube(h);
        for v in &mut m.vertices {
            *v += offset;
        }
        let (vol, com, i) = mass_properties(&m);
        let side = 2.0 * h;
        assert!((vol - side.powi(3)).abs() < 1e-12);
        assert!((com - offset).norm() < 1e-12);
        let expected = vol * side * side / 6.0;
        for k in 0..3 {
            assert!((i.0[4 * k] - expected).abs() < 1e-12, "diag {k}: {} vs {expected}", i.0[4 * k]);
        }
        assert!(i.0[1].abs() < 1e-12 && i.0[2].abs() < 1e-12 && i.0[5].abs() < 1e-12);
    }
}
