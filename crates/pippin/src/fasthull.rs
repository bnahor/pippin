//! Convex hull with hill-climbing support queries.
//!
//! parry's `ConvexPolyhedron` answers support queries by scanning every
//! vertex (and every face for support features). Mesh hulls of robot links
//! have hundreds to thousands of vertices, which made those scans the top
//! cost of manipulation scenes. This wrapper walks the hull's vertex graph
//! instead, starting from a precomputed seed near the query direction. On a
//! convex hull a greedy walk reaches the global maximum, so results are
//! exact; only the search is faster. Everything else delegates to parry.

use std::sync::atomic::{AtomicU32, Ordering};

use parry3d_f64::bounding_volume::{Aabb, BoundingSphere};
use parry3d_f64::mass_properties::MassProperties;
use parry3d_f64::math::{Pose, Real, Vector};
use parry3d_f64::query::{PointProjection, PointQuery, Ray, RayCast, RayIntersection};
use parry3d_f64::shape::{
    ConvexPolyhedron, FeatureId, PackedFeatureId, PolygonalFeature, PolygonalFeatureMap, Shape, ShapeType, SupportMap,
    TypedShape,
};

/// Hulls smaller than this are scanned directly (cheaper than walking).
const SCAN_BELOW: usize = 64;

#[derive(Debug)]
pub struct FastHull {
    inner: ConvexPolyhedron,
    /// Neighbouring vertices of each vertex (from hull edges).
    neighbors: Vec<Vec<u32>>,
    /// Seed directions and the extreme vertex along each.
    seeds: Vec<(Vector, u32)>,
    /// Last support vertex found. Consecutive queries (GJK/EPA iterations,
    /// successive steps) ask about nearby directions, so walks starting here
    /// are short. Shared across threads; a stale hint only costs steps.
    hint: AtomicU32,
}

impl Clone for FastHull {
    fn clone(&self) -> FastHull {
        FastHull {
            inner: self.inner.clone(),
            neighbors: self.neighbors.clone(),
            seeds: self.seeds.clone(),
            hint: AtomicU32::new(self.hint.load(Ordering::Relaxed)),
        }
    }
}

impl FastHull {
    pub fn new(inner: ConvexPolyhedron) -> FastHull {
        let n = inner.points().len();
        let mut neighbors = vec![Vec::new(); n];
        for e in inner.edges() {
            let [a, b] = e.vertices;
            neighbors[a as usize].push(b);
            neighbors[b as usize].push(a);
        }
        let mut seeds = Vec::new();
        for x in [-1.0, 0.0, 1.0] {
            for y in [-1.0, 0.0, 1.0] {
                for z in [-1.0, 0.0, 1.0] {
                    let d = Vector::new(x, y, z);
                    if d.length_squared() > 0.0 {
                        let d = d.normalize();
                        let best = (0..n).max_by(|&a, &b| inner.points()[a].dot(d).total_cmp(&inner.points()[b].dot(d))).unwrap_or(0);
                        seeds.push((d, best as u32));
                    }
                }
            }
        }
        FastHull { inner, neighbors, seeds, hint: AtomicU32::new(0) }
    }

    /// Index of the hull vertex extreme along `dir`.
    #[inline]
    fn support_vertex(&self, dir: Vector) -> usize {
        let pts = self.inner.points();
        if pts.len() < SCAN_BELOW {
            let (mut v, mut best) = (0, Real::NEG_INFINITY);
            for (i, p) in pts.iter().enumerate() {
                let d = p.dot(dir);
                if d > best {
                    best = d;
                    v = i;
                }
            }
            return v;
        }
        // start from the better of the last answer and the best seed
        let mut v = (self.hint.load(Ordering::Relaxed) as usize).min(pts.len() - 1);
        let (mut seed, mut seed_dot) = (0, Real::NEG_INFINITY);
        for &(d, i) in &self.seeds {
            let x = d.dot(dir);
            if x > seed_dot {
                seed_dot = x;
                seed = i as usize;
            }
        }
        if pts[seed].dot(dir) > pts[v].dot(dir) {
            v = seed;
        }
        let mut best = pts[v].dot(dir);
        loop {
            let mut next = v;
            for &nb in &self.neighbors[v] {
                let d = pts[nb as usize].dot(dir);
                if d > best {
                    best = d;
                    next = nb as usize;
                }
            }
            if next == v {
                self.hint.store(v as u32, Ordering::Relaxed);
                return v;
            }
            v = next;
        }
    }
}

impl SupportMap for FastHull {
    #[inline]
    fn local_support_point(&self, dir: Vector) -> Vector {
        self.inner.points()[self.support_vertex(dir)]
    }
}

impl PolygonalFeatureMap for FastHull {
    /// The face most aligned with `dir`, searched among the faces around the
    /// support vertex (the standard reference-face choice; parry scans all).
    fn local_support_feature(&self, dir: Vector, out: &mut PolygonalFeature) {
        let h = &self.inner;
        let v = &h.vertices()[self.support_vertex(dir)];
        let adj = &h.faces_adj_to_vertex()[v.first_adj_face_or_edge as usize..(v.first_adj_face_or_edge + v.num_adj_faces_or_edge) as usize];
        let fid = adj.iter().copied().max_by(|&a, &b| h.faces()[a as usize].normal.dot(dir).total_cmp(&h.faces()[b as usize].normal.dot(dir)));
        let Some(fid) = fid else {
            h.local_support_feature(dir, out);
            return;
        };
        let face = &h.faces()[fid as usize];
        let i1 = face.first_vertex_or_edge as usize;
        let num = (face.num_vertices_or_edges as usize).min(4);
        for i in 0..num {
            let vid = h.vertices_adj_to_face()[i1 + i];
            out.vertices[i] = h.points()[vid as usize];
            out.vids[i] = PackedFeatureId::vertex(vid);
            out.eids[i] = PackedFeatureId::edge(h.edges_adj_to_face()[i1 + i]);
        }
        out.fid = PackedFeatureId::face(fid);
        out.num_vertices = num;
    }

    fn is_convex_polyhedron(&self) -> bool {
        true
    }
}

impl RayCast for FastHull {
    fn cast_local_ray_and_get_normal(&self, ray: &Ray, max_time_of_impact: Real, solid: bool) -> Option<RayIntersection> {
        self.inner.cast_local_ray_and_get_normal(ray, max_time_of_impact, solid)
    }
}

impl PointQuery for FastHull {
    fn project_local_point(&self, pt: Vector, solid: bool) -> PointProjection {
        self.inner.project_local_point(pt, solid)
    }
    fn project_local_point_and_get_feature(&self, pt: Vector) -> (PointProjection, FeatureId) {
        self.inner.project_local_point_and_get_feature(pt)
    }
}

impl Shape for FastHull {
    fn compute_local_aabb(&self) -> Aabb {
        self.inner.compute_local_aabb()
    }
    fn compute_local_bounding_sphere(&self) -> BoundingSphere {
        self.inner.compute_local_bounding_sphere()
    }
    fn clone_dyn(&self) -> Box<dyn Shape> {
        Box::new(self.clone())
    }
    fn scale_dyn(&self, scale: Vector, _num_subdivisions: u32) -> Option<Box<dyn Shape>> {
        Some(Box::new(FastHull::new(self.inner.clone().scaled(scale)?)))
    }
    fn compute_aabb(&self, position: &Pose) -> Aabb {
        self.inner.compute_aabb(position)
    }
    fn mass_properties(&self, density: Real) -> MassProperties {
        self.inner.mass_properties(density)
    }
    fn is_convex(&self) -> bool {
        true
    }
    fn shape_type(&self) -> ShapeType {
        ShapeType::Custom
    }
    fn as_typed_shape(&self) -> TypedShape<'_> {
        TypedShape::Custom(self)
    }
    fn ccd_thickness(&self) -> Real {
        self.inner.ccd_thickness()
    }
    fn ccd_angular_thickness(&self) -> Real {
        self.inner.ccd_angular_thickness()
    }
    fn as_support_map(&self) -> Option<&dyn SupportMap> {
        Some(self as &dyn SupportMap)
    }
    fn as_polygonal_feature_map(&self) -> Option<(&dyn PolygonalFeatureMap, Real)> {
        Some((self as &dyn PolygonalFeatureMap, 0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hill_climb_matches_brute_force() {
        // points on a lumpy sphere: many vertices, all on the hull
        let pts: Vec<Vector> = (0..600)
            .map(|i| {
                let (t, p) = (i as Real * 2.399963, (1.0 - 2.0 * (i as Real + 0.5) / 600.0).acos());
                let r = 1.0 + 0.2 * (3.0 * t).sin() * (2.0 * p).cos();
                Vector::new(r * p.sin() * t.cos(), r * p.sin() * t.sin(), 0.7 * r * p.cos())
            })
            .collect();
        let hull = ConvexPolyhedron::from_convex_hull(&pts).unwrap();
        let fast = FastHull::new(hull.clone());
        for i in 0..2000 {
            let a = i as Real * 0.731;
            let d = Vector::new(a.sin() * (1.3 * a).cos(), (0.7 * a).cos(), (2.1 * a).sin()).normalize();
            let exact = hull.local_support_point(d).dot(d);
            let walked = fast.local_support_point(d).dot(d);
            assert!((exact - walked).abs() < 1e-12, "dir {d:?}: {exact} vs {walked}");
        }
    }
}
