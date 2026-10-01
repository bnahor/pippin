//! Triangle meshes for each geom, in geom-local coordinates, with the geom's
//! size baked in. All geoms share one vertex buffer tagged by geom id.

use pippin_env::{Scene, Shape};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Vtx {
    pub pos: [f32; 3],
    pub geom: u32,
    pub normal: [f32; 3],
    /// Bit 0: draw a checker pattern (planes).
    pub flags: u32,
}

const SLICES: usize = 24;
const STACKS: usize = 12;
/// Visual half-extent of planes declared infinite (size 0).
const PLANE_EXTENT: f32 = 20.0;

struct Builder {
    out: Vec<Vtx>,
    geom: u32,
    flags: u32,
}

impl Builder {
    fn tri(&mut self, p: [[f32; 3]; 3], n: [[f32; 3]; 3]) {
        for i in 0..3 {
            self.out.push(Vtx { pos: p[i], geom: self.geom, normal: n[i], flags: self.flags });
        }
    }
    fn flat(&mut self, a: [f32; 3], b: [f32; 3], c: [f32; 3]) {
        let n = normalize(cross(sub(b, a), sub(c, a)));
        self.tri([a, b, c], [n, n, n]);
    }
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn normalize(a: [f32; 3]) -> [f32; 3] {
    let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt().max(1e-12);
    [a[0] / n, a[1] / n, a[2] / n]
}

/// Surface of revolution around z from a profile of (radius, z, normal_r, normal_z).
fn revolve(b: &mut Builder, profile: &[(f32, f32, f32, f32)]) {
    let ring = |k: usize| {
        let t = 2.0 * std::f32::consts::PI * (k % SLICES) as f32 / SLICES as f32;
        (t.cos(), t.sin())
    };
    for w in profile.windows(2) {
        let (r0, z0, nr0, nz0) = w[0];
        let (r1, z1, nr1, nz1) = w[1];
        for k in 0..SLICES {
            let (c0, s0) = ring(k);
            let (c1, s1) = ring(k + 1);
            let p = |r: f32, z: f32, c: f32, s: f32| [r * c, r * s, z];
            let n = |nr: f32, nz: f32, c: f32, s: f32| normalize([nr * c, nr * s, nz]);
            let (a, bb, c, d) = (p(r0, z0, c0, s0), p(r0, z0, c1, s1), p(r1, z1, c1, s1), p(r1, z1, c0, s0));
            let (na, nb, nc, nd) = (n(nr0, nz0, c0, s0), n(nr0, nz0, c1, s1), n(nr1, nz1, c1, s1), n(nr1, nz1, c0, s0));
            b.tri([a, bb, c], [na, nb, nc]);
            b.tri([a, c, d], [na, nc, nd]);
        }
    }
}

fn sphere_profile(r: f32, z_offset: f32, from: f32, to: f32) -> Vec<(f32, f32, f32, f32)> {
    // polar angle from `from` to `to` (0 = +z pole)
    (0..=STACKS)
        .map(|i| {
            let phi = from + (to - from) * i as f32 / STACKS as f32;
            let (s, c) = phi.sin_cos();
            (r * s, z_offset + r * c, s, c)
        })
        .collect()
}

fn box_mesh(b: &mut Builder, h: [f32; 3]) {
    for axis in 0..3 {
        for sign in [-1.0f32, 1.0] {
            let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
            let corner = |su: f32, sv: f32| {
                let mut p = [0.0; 3];
                p[axis] = sign * h[axis];
                p[u] = su * h[u];
                p[v] = sv * h[v];
                p
            };
            let (a, bb, c, d) = (corner(-1.0, -1.0), corner(1.0, -1.0), corner(1.0, 1.0), corner(-1.0, 1.0));
            if sign > 0.0 {
                b.flat(a, bb, c);
                b.flat(a, c, d);
            } else {
                b.flat(a, c, bb);
                b.flat(a, d, c);
            }
        }
    }
}

/// Indexed geometry: unique vertices plus triangle indices. Smooth surfaces
/// share vertices between triangles, so the GPU's post-transform cache runs
/// the vertex shader far fewer times than for a raw triangle list.
pub struct IndexedMesh {
    pub vertices: Vec<Vtx>,
    pub indices: Vec<u32>,
}

/// `lod_error`: simplification error allowed for meshes, relative to each
/// mesh's extent (0 keeps full detail). Renders at ~N pixels can use ~1/N.
pub fn build_indexed(scene: &Scene, lod_error: f32) -> IndexedMesh {
    let tris = build(scene);
    let mut map = std::collections::HashMap::with_capacity(tris.len());
    let mut vertices = Vec::new();
    let mut indices: Vec<u32> = tris
        .iter()
        .map(|v| {
            let key = (v.pos.map(f32::to_bits), v.normal.map(f32::to_bits), v.geom, v.flags);
            *map.entry(key).or_insert_with(|| {
                vertices.push(*v);
                (vertices.len() - 1) as u32
            })
        })
        .collect();
    for (g, geom) in scene.geoms.iter().enumerate() {
        if let (Shape::Mesh(k), true) = (geom.shape, visible(geom)) {
            if let Some(mesh) = scene.meshes.get(k) {
                append_mesh(mesh, g as u32, lod_error, &mut vertices, &mut indices);
            }
        }
    }
    let indices = if vertices.is_empty() { indices } else { meshopt::optimize_vertex_cache(&indices, vertices.len()) };
    IndexedMesh { vertices, indices }
}

/// MuJoCo convention: groups 0-2 are visible by default.
fn visible(geom: &pippin_env::GeomVisual) -> bool {
    geom.group <= 2 && geom.rgba[3] > 0.0
}

/// Weld, simplify to `lod_error`, and append a mesh with smooth normals.
fn append_mesh(mesh: &pippin_env::Mesh, geom: u32, lod_error: f32, vertices: &mut Vec<Vtx>, indices: &mut Vec<u32>) {
    // weld identical positions so neighbouring triangles share vertices
    let mut map = std::collections::HashMap::new();
    let mut pos: Vec<[f32; 3]> = vec![];
    let mut idx: Vec<u32> = Vec::with_capacity(mesh.triangles.len() * 3);
    for t in &mesh.triangles {
        for &i in t {
            let p = mesh.vertices[i as usize];
            let id = *map.entry(p.map(f32::to_bits)).or_insert_with(|| {
                pos.push(p);
                (pos.len() - 1) as u32
            });
            idx.push(id);
        }
    }
    if lod_error > 0.0 && idx.len() > 3 * 64 {
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(pos.as_ptr() as *const u8, pos.len() * 12) };
        if let Ok(adapter) = meshopt::VertexDataAdapter::new(bytes, 12, 0) {
            let simplified = meshopt::simplify(&idx, &adapter, 3 * 32, lod_error, meshopt::SimplifyOptions::None, None);
            if simplified.len() >= 3 {
                idx = simplified;
            }
        }
    }
    // area-weighted vertex normals
    let mut nrm = vec![[0.0f32; 3]; pos.len()];
    for t in idx.chunks(3) {
        let [a, b, c] = [pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]];
        let n = cross(sub(b, a), sub(c, a));
        for &i in t {
            for k in 0..3 {
                nrm[i as usize][k] += n[k];
            }
        }
    }
    let base = vertices.len() as u32;
    vertices.extend(pos.iter().zip(&nrm).map(|(&p, &n)| Vtx { pos: p, geom, normal: normalize(n), flags: 0 }));
    indices.extend(idx.iter().map(|i| base + i));
}

/// Triangle list (three vertices per triangle).
pub fn build(scene: &Scene) -> Vec<Vtx> {
    let mut b = Builder { out: vec![], geom: 0, flags: 0 };
    for (g, geom) in scene.geoms.iter().enumerate() {
        // MuJoCo convention: groups 0-2 are visible by default
        if geom.group > 2 || geom.rgba[3] <= 0.0 {
            continue;
        }
        b.geom = g as u32;
        b.flags = matches!(geom.shape, Shape::Plane) as u32;
        let s = geom.size;
        match geom.shape {
            Shape::Plane => {
                let ex = if s[0] > 0.0 { s[0] } else { PLANE_EXTENT };
                let ey = if s[1] > 0.0 { s[1] } else { PLANE_EXTENT };
                b.flat([-ex, -ey, 0.0], [ex, -ey, 0.0], [ex, ey, 0.0]);
                b.flat([-ex, -ey, 0.0], [ex, ey, 0.0], [-ex, ey, 0.0]);
            }
            Shape::Sphere => revolve(&mut b, &sphere_profile(s[0], 0.0, 0.0, std::f32::consts::PI)),
            Shape::Ellipsoid => {
                let start = b.out.len();
                revolve(&mut b, &sphere_profile(1.0, 0.0, 0.0, std::f32::consts::PI));
                for v in &mut b.out[start..] {
                    for i in 0..3 {
                        v.pos[i] *= s[i];
                        v.normal[i] /= s[i];
                    }
                    v.normal = normalize(v.normal);
                }
            }
            Shape::Capsule => {
                let (r, h) = (s[0], s[1]);
                let half = std::f32::consts::FRAC_PI_2;
                let mut prof = sphere_profile(r, h, 0.0, half);
                prof.extend(sphere_profile(r, -h, half, std::f32::consts::PI));
                revolve(&mut b, &prof);
            }
            Shape::Cylinder => {
                let (r, h) = (s[0], s[1]);
                revolve(&mut b, &[(0.0, h, 0.0, 1.0), (r, h, 0.0, 1.0)]);
                revolve(&mut b, &[(r, h, 1.0, 0.0), (r, -h, 1.0, 0.0)]);
                revolve(&mut b, &[(r, -h, 0.0, -1.0), (0.0, -h, 0.0, -1.0)]);
            }
            Shape::Box => box_mesh(&mut b, s),
            // meshes are appended indexed and simplified by `build_indexed`
            Shape::Mesh(_) => {}
        }
    }
    b.out
}
