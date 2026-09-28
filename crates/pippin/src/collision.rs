//! Narrow-phase collision between primitive shapes.
//!
//! Convention: every routine returns contacts whose normal points from the
//! first geom toward the second, with `depth > 0` meaning penetration.

use crate::math::{Mat3, Real, Vec3};
use crate::model::GeomType;

#[derive(Clone, Copy, Debug)]
pub struct Contact {
    pub pos: Vec3,
    pub normal: Vec3,
    pub depth: Real,
    pub geom: [usize; 2],
    /// Stable per-pair index used to match contacts across steps (warm start).
    pub feature: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct GeomPose {
    pub typ: GeomType,
    pub pos: Vec3,
    pub mat: Mat3,
    pub size: Vec3,
}

/// Raw contact point before geom ids are attached.
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub pos: Vec3,
    pub normal: Vec3,
    pub depth: Real,
}

/// Collide two geoms; pushes hits (normal from `a` to `b`) into `out`.
pub fn collide(a: &GeomPose, b: &GeomPose, margin: Real, out: &mut Vec<Hit>) {
    use GeomType::*;
    // order so that type(a) <= type(b); flip normals if swapped
    if (a.typ as u8) > (b.typ as u8) {
        let start = out.len();
        collide(b, a, margin, out);
        for h in &mut out[start..] {
            h.normal = -h.normal;
        }
        return;
    }
    match (a.typ, b.typ) {
        (Plane, Sphere) => plane_sphere(a, b.pos, b.size[0], margin, out),
        (Plane, Capsule) => {
            let (p0, p1) = segment(b);
            plane_sphere(a, p0, b.size[0], margin, out);
            plane_sphere(a, p1, b.size[0], margin, out);
        }
        (Plane, Box) => plane_box(a, b, margin, out),
        (Plane, Cylinder) => plane_cylinder(a, b, margin, out),
        (Sphere, Sphere) => sphere_sphere(a.pos, a.size[0], b.pos, b.size[0], margin, out),
        (Sphere, Capsule) => {
            let (p0, p1) = segment(b);
            let q = closest_on_segment(p0, p1, a.pos);
            sphere_sphere(a.pos, a.size[0], q, b.size[0], margin, out);
        }
        (Sphere, Box) => sphere_box(a.pos, a.size[0], b, margin, out),
        (Capsule, Capsule) => capsule_capsule(a, b, margin, out),
        (Capsule, Box) => capsule_box(a, b, margin, out),
        (Box, Box) => box_box(a, b, margin, out),
        _ => {} // unsupported pairs are filtered at compile time
    }
}

pub fn pair_supported(a: GeomType, b: GeomType) -> bool {
    use GeomType::*;
    let (a, b) = if (a as u8) <= (b as u8) { (a, b) } else { (b, a) };
    match (a, b) {
        (Plane, Plane) => false,
        (Plane, _) => true,
        (_, Cylinder) => false,
        _ => true,
    }
}

fn segment(g: &GeomPose) -> (Vec3, Vec3) {
    let ax = g.mat.col(2) * g.size[1];
    (g.pos - ax, g.pos + ax)
}

fn closest_on_segment(a: Vec3, b: Vec3, p: Vec3) -> Vec3 {
    let d = b - a;
    let l2 = d.norm2();
    if l2 < 1e-18 {
        return a;
    }
    let t = ((p - a).dot(d) / l2).clamp(0.0, 1.0);
    a + d * t
}

/// Closest points between segments p0-p1 and q0-q1.
fn closest_segments(p0: Vec3, p1: Vec3, q0: Vec3, q1: Vec3) -> (Vec3, Vec3) {
    let d1 = p1 - p0;
    let d2 = q1 - q0;
    let r = p0 - q0;
    let a = d1.norm2();
    let e = d2.norm2();
    let f = d2.dot(r);
    let (s, t);
    if a < 1e-18 && e < 1e-18 {
        return (p0, q0);
    }
    if a < 1e-18 {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(r);
        if e < 1e-18 {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = d1.dot(d2);
            let denom = a * e - b * b;
            let mut s0 = if denom > 1e-18 { ((b * f - c * e) / denom).clamp(0.0, 1.0) } else { 0.0 };
            let mut t0 = (b * s0 + f) / e;
            if t0 < 0.0 {
                t0 = 0.0;
                s0 = (-c / a).clamp(0.0, 1.0);
            } else if t0 > 1.0 {
                t0 = 1.0;
                s0 = ((b - c) / a).clamp(0.0, 1.0);
            }
            s = s0;
            t = t0;
        }
    }
    (p0 + d1 * s, q0 + d2 * t)
}

fn plane_sphere(plane: &GeomPose, c: Vec3, r: Real, margin: Real, out: &mut Vec<Hit>) {
    let n = plane.mat.col(2);
    let dist = (c - plane.pos).dot(n);
    let depth = r - dist;
    if depth > -margin {
        // midpoint between sphere surface and plane
        let pos = c - n * (dist + r) * 0.5;
        out.push(Hit { pos, normal: n, depth });
    }
}

fn plane_box(plane: &GeomPose, b: &GeomPose, margin: Real, out: &mut Vec<Hit>) {
    let n = plane.mat.col(2);
    let mut hits: [(Real, Vec3); 8] = [(0.0, Vec3::ZERO); 8];
    let mut k = 0;
    for i in 0..8 {
        let s = Vec3::new(
            if i & 1 == 0 { -1.0 } else { 1.0 },
            if i & 2 == 0 { -1.0 } else { 1.0 },
            if i & 4 == 0 { -1.0 } else { 1.0 },
        );
        let corner = b.pos + b.mat.mul_vec(b.size.mul_elem(s));
        let dist = (corner - plane.pos).dot(n);
        if dist < margin {
            hits[k] = (dist, corner);
            k += 1;
        }
    }
    let hits = &mut hits[..k];
    hits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    for &(dist, corner) in hits.iter().take(4) {
        out.push(Hit { pos: corner - n * (dist * 0.5), normal: n, depth: -dist });
    }
}

fn plane_cylinder(plane: &GeomPose, c: &GeomPose, margin: Real, out: &mut Vec<Hit>) {
    let n = plane.mat.col(2);
    let axis = c.mat.col(2);
    let (r, h) = (c.size[0], c.size[1]);
    let radial = n - axis * axis.dot(n);
    let rl = radial.norm();
    let mut pts: Vec<Vec3> = Vec::with_capacity(8);
    for cap in [-1.0, 1.0] {
        let cc = c.pos + axis * (h * cap);
        if rl > 0.05 {
            // deepest rim point, and its two neighbours for a stable line contact
            let d = radial * (-1.0 / rl);
            pts.push(cc + d * r);
        } else {
            let (t1, t2) = axis.tangents();
            for v in [t1, -t1, t2, -t2] {
                pts.push(cc + v * r);
            }
        }
    }
    let mut hits: Vec<(Real, Vec3)> = pts
        .into_iter()
        .map(|p| ((p - plane.pos).dot(n), p))
        .filter(|(d, _)| *d < margin)
        .collect();
    hits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    for &(dist, p) in hits.iter().take(4) {
        out.push(Hit { pos: p - n * (dist * 0.5), normal: n, depth: -dist });
    }
}

fn sphere_sphere(c1: Vec3, r1: Real, c2: Vec3, r2: Real, margin: Real, out: &mut Vec<Hit>) {
    let d = c2 - c1;
    let dist = d.norm();
    let depth = r1 + r2 - dist;
    if depth <= -margin {
        return;
    }
    let n = if dist > 1e-12 { d * (1.0 / dist) } else { Vec3::Z };
    let pos = c1 + n * (r1 - depth * 0.5);
    out.push(Hit { pos, normal: n, depth });
}

/// Sphere vs box. Normal points from sphere to box.
fn sphere_box(c: Vec3, r: Real, b: &GeomPose, margin: Real, out: &mut Vec<Hit>) {
    let local = b.mat.tmul_vec(c - b.pos);
    let h = b.size;
    let clamped = Vec3::new(local[0].clamp(-h[0], h[0]), local[1].clamp(-h[1], h[1]), local[2].clamp(-h[2], h[2]));
    let diff = local - clamped;
    let dist = diff.norm();
    if dist > 1e-12 {
        let depth = r - dist;
        if depth <= -margin {
            return;
        }
        // normal from sphere to box = -(diff / dist)
        let n_local = diff * (-1.0 / dist);
        let n = b.mat.mul_vec(n_local);
        let surf = b.pos + b.mat.mul_vec(clamped);
        out.push(Hit { pos: surf - n * (depth * 0.5), normal: n, depth });
    } else {
        // center inside the box: push out through the nearest face
        let mut best = (Real::INFINITY, 0, 1.0);
        for i in 0..3 {
            for s in [-1.0, 1.0] {
                let d = h[i] - s * local[i];
                if d < best.0 {
                    best = (d, i, s);
                }
            }
        }
        let (d, i, s) = best;
        let mut nl = Vec3::ZERO;
        nl[i] = -s; // from sphere into box
        let n = b.mat.mul_vec(nl);
        let depth = r + d;
        out.push(Hit { pos: c + n * (r - depth * 0.5), normal: n, depth });
    }
}

fn capsule_capsule(a: &GeomPose, b: &GeomPose, margin: Real, out: &mut Vec<Hit>) {
    let (p0, p1) = segment(a);
    let (q0, q1) = segment(b);
    let (ra, rb) = (a.size[0], b.size[0]);
    let (x, y) = closest_segments(p0, p1, q0, q1);
    sphere_sphere(x, ra, y, rb, margin, out);
    // near-parallel capsules: add endpoint contacts for a stable line contact
    let da = (p1 - p0).normalized();
    let db = (q1 - q0).normalized();
    if da.cross(db).norm() < 0.05 {
        for p in [p0, p1] {
            let q = closest_on_segment(q0, q1, p);
            if (q - y).norm() > 0.25 * (ra + rb) || (p - x).norm() > 0.25 * (ra + rb) {
                sphere_sphere(p, ra, q, rb, margin, out);
            }
        }
    }
}

/// Signed distance from a point to a box (negative inside).
fn box_sdf(b: &GeomPose, p: Vec3) -> Real {
    let l = b.mat.tmul_vec(p - b.pos);
    let q = l.abs() - b.size;
    let outside = q.max(Vec3::ZERO).norm();
    let inside = q[0].max(q[1]).max(q[2]).min(0.0);
    outside + inside
}

fn capsule_box(a: &GeomPose, b: &GeomPose, margin: Real, out: &mut Vec<Hit>) {
    let (p0, p1) = segment(a);
    let r = a.size[0];
    // the box SDF is convex along the segment: golden-section for the minimum
    let f = |t: Real| box_sdf(b, p0 + (p1 - p0) * t);
    let g = 0.618_033_988_75;
    let (mut lo, mut hi) = (0.0, 1.0);
    let (mut x1, mut x2) = (hi - g * (hi - lo), lo + g * (hi - lo));
    let (mut f1, mut f2) = (f(x1), f(x2));
    for _ in 0..30 {
        if f1 < f2 {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = hi - g * (hi - lo);
            f1 = f(x1);
        } else {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = lo + g * (hi - lo);
            f2 = f(x2);
        }
    }
    let tmin = 0.5 * (lo + hi);
    let mut ts = vec![tmin];
    for t in [0.0, 1.0] {
        if (t - tmin).abs() > 0.1 && f(t) < r + margin {
            ts.push(t);
        }
    }
    for t in ts {
        sphere_box(p0 + (p1 - p0) * t, r, b, margin, out);
    }
}

/// Box-box via the separating axis test with face clipping (up to 4 points).
fn box_box(a: &GeomPose, b: &GeomPose, margin: Real, out: &mut Vec<Hit>) {
    let d = b.pos - a.pos;
    let ax = [a.mat.col(0), a.mat.col(1), a.mat.col(2)];
    let bx = [b.mat.col(0), b.mat.col(1), b.mat.col(2)];
    let proj = |axes: &[Vec3; 3], h: Vec3, l: Vec3| -> Real {
        h[0] * axes[0].dot(l).abs() + h[1] * axes[1].dot(l).abs() + h[2] * axes[2].dot(l).abs()
    };

    // (overlap, axis, kind) where kind 0..3 = face of a, 3..6 = face of b, 6.. = edge pair
    let mut best: Option<(Real, Vec3, usize)> = None;
    let mut best_face: Option<(Real, Vec3, usize)> = None;
    for k in 0..15 {
        let l = if k < 3 {
            ax[k]
        } else if k < 6 {
            bx[k - 3]
        } else {
            let c = ax[(k - 6) / 3].cross(bx[(k - 6) % 3]);
            let n = c.norm();
            if n < 1e-6 {
                continue;
            }
            c * (1.0 / n)
        };
        let overlap = proj(&ax, a.size, l) + proj(&bx, b.size, l) - d.dot(l).abs();
        if overlap < -margin {
            return;
        }
        let l = if d.dot(l) < 0.0 { -l } else { l };
        if best.is_none_or(|(o, _, _)| overlap < o) {
            best = Some((overlap, l, k));
        }
        if k < 6 && best_face.is_none_or(|(o, _, _)| overlap < o) {
            best_face = Some((overlap, l, k));
        }
    }
    let (Some(mut best), Some(face)) = (best, best_face) else { return };
    // prefer face contacts unless an edge axis is clearly better (stability)
    if best.2 >= 6 && face.0 < best.0 * 1.05 + 1e-4 {
        best = face;
    }
    let (overlap, n, k) = best;

    if k >= 6 {
        let (i, j) = ((k - 6) / 3, (k - 6) % 3);
        let mut pa = a.pos;
        for m in 0..3 {
            if m != i {
                pa += ax[m] * (a.size[m] * ax[m].dot(n).signum());
            }
        }
        let mut pb = b.pos;
        for m in 0..3 {
            if m != j {
                pb -= bx[m] * (b.size[m] * bx[m].dot(n).signum());
            }
        }
        let (x, y) = closest_segments(
            pa - ax[i] * a.size[i],
            pa + ax[i] * a.size[i],
            pb - bx[j] * b.size[j],
            pb + bx[j] * b.size[j],
        );
        out.push(Hit { pos: (x + y) * 0.5, normal: n, depth: overlap });
        return;
    }

    // face contact: reference box owns the separating face
    let (rf, inc, raxes, iaxes, nref, fi) = if k < 3 { (a, b, ax, bx, n, k) } else { (b, a, bx, ax, -n, k - 3) };
    // nref points from the reference box toward the incident box
    let ref_center = rf.pos + nref * rf.size[fi];
    let (u, v) = ((fi + 1) % 3, (fi + 2) % 3);
    // incident face: most anti-parallel to nref
    let mut j = 0;
    let mut bestdot = Real::INFINITY;
    for m in 0..3 {
        let dt = iaxes[m].dot(nref);
        if -dt.abs() < bestdot {
            bestdot = -dt.abs();
            j = m;
        }
    }
    let s = -iaxes[j].dot(nref).signum();
    let ic = inc.pos + iaxes[j] * (s * inc.size[j]);
    let (ku, kv) = ((j + 1) % 3, (j + 2) % 3);
    let eu = iaxes[ku] * inc.size[ku];
    let ev = iaxes[kv] * inc.size[kv];
    let mut poly: Vec<Vec3> = vec![ic + eu + ev, ic - eu + ev, ic - eu - ev, ic + eu - ev];

    // clip against the 4 side planes of the reference face
    for (axis, ext) in [(raxes[u], rf.size[u]), (raxes[v], rf.size[v])] {
        for sign in [1.0, -1.0] {
            let pn = axis * sign;
            let off = pn.dot(ref_center) + ext;
            poly = clip(&poly, pn, off);
            if poly.is_empty() {
                return;
            }
        }
    }

    let mut pts: Vec<(Real, Vec3)> = poly
        .into_iter()
        .map(|p| ((p - ref_center).dot(nref), p))
        .filter(|(sep, _)| *sep < margin)
        .collect();
    reduce_to_four(&mut pts);
    for (sep, p) in pts {
        out.push(Hit { pos: p - nref * (sep * 0.5), normal: n, depth: -sep });
    }
}

/// Sutherland-Hodgman: keep the part of `poly` with pn.p <= off.
fn clip(poly: &[Vec3], pn: Vec3, off: Real) -> Vec<Vec3> {
    let mut out = Vec::with_capacity(poly.len() + 2);
    for i in 0..poly.len() {
        let a = poly[i];
        let b = poly[(i + 1) % poly.len()];
        let da = pn.dot(a) - off;
        let db = pn.dot(b) - off;
        if da <= 0.0 {
            out.push(a);
        }
        if (da < 0.0) != (db < 0.0) && (da - db).abs() > 1e-15 {
            out.push(a + (b - a) * (da / (da - db)));
        }
    }
    out
}

/// Keep the deepest point plus the three that best span the contact patch.
fn reduce_to_four(pts: &mut Vec<(Real, Vec3)>) {
    if pts.len() <= 4 {
        return;
    }
    let mut keep: Vec<(Real, Vec3)> = Vec::with_capacity(4);
    let deepest = pts.iter().copied().min_by(|a, b| a.0.partial_cmp(&b.0).unwrap()).unwrap();
    keep.push(deepest);
    let far = pts.iter().copied().max_by(|a, b| (a.1 - deepest.1).norm2().partial_cmp(&(b.1 - deepest.1).norm2()).unwrap()).unwrap();
    keep.push(far);
    let area = |p: Vec3| (keep[1].1 - keep[0].1).cross(p - keep[0].1).norm();
    let third = pts.iter().copied().max_by(|a, b| area(a.1).partial_cmp(&area(b.1)).unwrap()).unwrap();
    keep.push(third);
    // fourth: maximize distance to the triangle's vertices sum (cheap spread proxy)
    let spread = |p: Vec3| keep.iter().map(|k| (k.1 - p).norm()).fold(Real::INFINITY, Real::min);
    let fourth = pts.iter().copied().max_by(|a, b| spread(a.1).partial_cmp(&spread(b.1)).unwrap()).unwrap();
    keep.push(fourth);
    *pts = keep;
}
