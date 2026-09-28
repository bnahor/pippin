//! Minimal fixed-size math. Deliberately plain-old-data so every routine here
//! has a line-for-line counterpart in the Metal kernels.

use std::ops::{Add, AddAssign, Index, IndexMut, Mul, Neg, Sub, SubAssign};

pub type Real = f64;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3(pub [Real; 3]);

impl Vec3 {
    pub const ZERO: Vec3 = Vec3([0.0; 3]);
    pub const X: Vec3 = Vec3([1.0, 0.0, 0.0]);
    pub const Y: Vec3 = Vec3([0.0, 1.0, 0.0]);
    pub const Z: Vec3 = Vec3([0.0, 0.0, 1.0]);

    #[inline]
    pub const fn new(x: Real, y: Real, z: Real) -> Self {
        Vec3([x, y, z])
    }
    #[inline]
    pub fn dot(self, o: Vec3) -> Real {
        self.0[0] * o.0[0] + self.0[1] * o.0[1] + self.0[2] * o.0[2]
    }
    #[inline]
    pub fn cross(self, o: Vec3) -> Vec3 {
        let [a, b, c] = self.0;
        let [x, y, z] = o.0;
        Vec3([b * z - c * y, c * x - a * z, a * y - b * x])
    }
    #[inline]
    pub fn norm2(self) -> Real {
        self.dot(self)
    }
    #[inline]
    pub fn norm(self) -> Real {
        self.norm2().sqrt()
    }
    #[inline]
    pub fn normalized(self) -> Vec3 {
        let n = self.norm();
        if n > 1e-15 {
            self * (1.0 / n)
        } else {
            Vec3::ZERO
        }
    }
    #[inline]
    pub fn abs(self) -> Vec3 {
        Vec3(self.0.map(Real::abs))
    }
    #[inline]
    pub fn mul_elem(self, o: Vec3) -> Vec3 {
        Vec3([self.0[0] * o.0[0], self.0[1] * o.0[1], self.0[2] * o.0[2]])
    }
    #[inline]
    pub fn min(self, o: Vec3) -> Vec3 {
        Vec3([self.0[0].min(o.0[0]), self.0[1].min(o.0[1]), self.0[2].min(o.0[2])])
    }
    #[inline]
    pub fn max(self, o: Vec3) -> Vec3 {
        Vec3([self.0[0].max(o.0[0]), self.0[1].max(o.0[1]), self.0[2].max(o.0[2])])
    }
    /// Two unit vectors completing an orthonormal frame with `self` (assumed unit).
    pub fn tangents(self) -> (Vec3, Vec3) {
        let a = if self.0[0].abs() < 0.57 { Vec3::X } else { Vec3::Y };
        let t1 = self.cross(a).normalized();
        let t2 = self.cross(t1);
        (t1, t2)
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    #[inline]
    fn add(self, o: Vec3) -> Vec3 {
        Vec3([self.0[0] + o.0[0], self.0[1] + o.0[1], self.0[2] + o.0[2]])
    }
}
impl Sub for Vec3 {
    type Output = Vec3;
    #[inline]
    fn sub(self, o: Vec3) -> Vec3 {
        Vec3([self.0[0] - o.0[0], self.0[1] - o.0[1], self.0[2] - o.0[2]])
    }
}
impl Neg for Vec3 {
    type Output = Vec3;
    #[inline]
    fn neg(self) -> Vec3 {
        Vec3(self.0.map(|v| -v))
    }
}
impl Mul<Real> for Vec3 {
    type Output = Vec3;
    #[inline]
    fn mul(self, s: Real) -> Vec3 {
        Vec3(self.0.map(|v| v * s))
    }
}
impl AddAssign for Vec3 {
    #[inline]
    fn add_assign(&mut self, o: Vec3) {
        *self = *self + o;
    }
}
impl SubAssign for Vec3 {
    #[inline]
    fn sub_assign(&mut self, o: Vec3) {
        *self = *self - o;
    }
}
impl Index<usize> for Vec3 {
    type Output = Real;
    #[inline]
    fn index(&self, i: usize) -> &Real {
        &self.0[i]
    }
}
impl IndexMut<usize> for Vec3 {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut Real {
        &mut self.0[i]
    }
}

/// Row-major 3x3 matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat3(pub [Real; 9]);

impl Default for Mat3 {
    fn default() -> Self {
        Mat3::IDENTITY
    }
}

impl Mat3 {
    pub const IDENTITY: Mat3 = Mat3([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    pub const ZERO: Mat3 = Mat3([0.0; 9]);

    pub fn diag(d: Vec3) -> Mat3 {
        Mat3([d[0], 0.0, 0.0, 0.0, d[1], 0.0, 0.0, 0.0, d[2]])
    }
    pub fn from_cols(a: Vec3, b: Vec3, c: Vec3) -> Mat3 {
        Mat3([a[0], b[0], c[0], a[1], b[1], c[1], a[2], b[2], c[2]])
    }
    #[inline]
    pub fn col(&self, j: usize) -> Vec3 {
        Vec3([self.0[j], self.0[3 + j], self.0[6 + j]])
    }
    #[inline]
    pub fn mul_vec(&self, v: Vec3) -> Vec3 {
        let m = &self.0;
        Vec3([
            m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
            m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
            m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
        ])
    }
    #[inline]
    pub fn tmul_vec(&self, v: Vec3) -> Vec3 {
        let m = &self.0;
        Vec3([
            m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
            m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
            m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
        ])
    }
    pub fn mul_mat(&self, o: &Mat3) -> Mat3 {
        let mut r = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                r[3 * i + j] = (0..3).map(|k| self.0[3 * i + k] * o.0[3 * k + j]).sum();
            }
        }
        Mat3(r)
    }
    pub fn transpose(&self) -> Mat3 {
        let m = &self.0;
        Mat3([m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]])
    }
    pub fn skew(v: Vec3) -> Mat3 {
        Mat3([0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0])
    }
    pub fn scale(&self, s: Real) -> Mat3 {
        Mat3(self.0.map(|v| v * s))
    }
    pub fn add(&self, o: &Mat3) -> Mat3 {
        let mut r = self.0;
        for (a, b) in r.iter_mut().zip(o.0.iter()) {
            *a += b;
        }
        Mat3(r)
    }
    /// R * self * R^T
    pub fn rotate(&self, r: &Mat3) -> Mat3 {
        r.mul_mat(self).mul_mat(&r.transpose())
    }
}

/// Unit quaternion, (w, x, y, z) like MuJoCo.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat(pub [Real; 4]);

impl Default for Quat {
    fn default() -> Self {
        Quat::IDENTITY
    }
}

impl Quat {
    pub const IDENTITY: Quat = Quat([1.0, 0.0, 0.0, 0.0]);

    pub fn from_axis_angle(axis: Vec3, angle: Real) -> Quat {
        let a = axis.normalized();
        let (s, c) = (0.5 * angle).sin_cos();
        Quat([c, a[0] * s, a[1] * s, a[2] * s])
    }
    pub fn normalized(self) -> Quat {
        let n = self.0.iter().map(|v| v * v).sum::<Real>().sqrt();
        if n < 1e-15 {
            return Quat::IDENTITY;
        }
        Quat(self.0.map(|v| v / n))
    }
    pub fn conj(self) -> Quat {
        let [w, x, y, z] = self.0;
        Quat([w, -x, -y, -z])
    }
    pub fn mul(self, o: Quat) -> Quat {
        let [a, b, c, d] = self.0;
        let [e, f, g, h] = o.0;
        Quat([
            a * e - b * f - c * g - d * h,
            a * f + b * e + c * h - d * g,
            a * g - b * h + c * e + d * f,
            a * h + b * g - c * f + d * e,
        ])
    }
    pub fn rotate(self, v: Vec3) -> Vec3 {
        self.to_mat().mul_vec(v)
    }
    pub fn to_mat(self) -> Mat3 {
        let [w, x, y, z] = self.0;
        let (ww, xx, yy, zz) = (w * w, x * x, y * y, z * z);
        let (xy, xz, yz, wx, wy, wz) = (x * y, x * z, y * z, w * x, w * y, w * z);
        Mat3([
            ww + xx - yy - zz,
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            ww - xx + yy - zz,
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            ww - xx - yy + zz,
        ])
    }
    /// Rotation matrix to quaternion (Shepperd's method).
    pub fn from_mat(m: &Mat3) -> Quat {
        let m = &m.0;
        let tr = m[0] + m[4] + m[8];
        let q = if tr > 0.0 {
            let s = (tr + 1.0).sqrt() * 2.0;
            [0.25 * s, (m[7] - m[5]) / s, (m[2] - m[6]) / s, (m[3] - m[1]) / s]
        } else if m[0] > m[4] && m[0] > m[8] {
            let s = (1.0 + m[0] - m[4] - m[8]).sqrt() * 2.0;
            [(m[7] - m[5]) / s, 0.25 * s, (m[1] + m[3]) / s, (m[2] + m[6]) / s]
        } else if m[4] > m[8] {
            let s = (1.0 + m[4] - m[0] - m[8]).sqrt() * 2.0;
            [(m[2] - m[6]) / s, (m[1] + m[3]) / s, 0.25 * s, (m[5] + m[7]) / s]
        } else {
            let s = (1.0 + m[8] - m[0] - m[4]).sqrt() * 2.0;
            [(m[3] - m[1]) / s, (m[2] + m[6]) / s, (m[5] + m[7]) / s, 0.25 * s]
        };
        let q = Quat(q).normalized();
        if q.0[0] < 0.0 {
            Quat(q.0.map(|v| -v))
        } else {
            q
        }
    }
    /// Rotation taking +z onto `dir`.
    pub fn from_z_to(dir: Vec3) -> Quat {
        let d = dir.normalized();
        let c = Vec3::Z.dot(d);
        if c < -1.0 + 1e-12 {
            return Quat([0.0, 1.0, 0.0, 0.0]);
        }
        let axis = Vec3::Z.cross(d);
        Quat([1.0 + c, axis[0], axis[1], axis[2]]).normalized()
    }
    /// q * exp(omega * h / 2): integrate a body-frame angular velocity (MuJoCo convention).
    pub fn integrate_local(self, omega: Vec3, h: Real) -> Quat {
        let n = omega.norm();
        if n * h < 1e-15 {
            return self;
        }
        self.mul(Quat::from_axis_angle(omega * (1.0 / n), n * h)).normalized()
    }
}

/// Spatial motion/force vector [angular; linear], expressed in world axes about
/// the world origin. Composite quantities can simply be summed in this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Spatial {
    pub ang: Vec3,
    pub lin: Vec3,
}

impl Spatial {
    pub const ZERO: Spatial = Spatial { ang: Vec3::ZERO, lin: Vec3::ZERO };

    #[inline]
    pub fn new(ang: Vec3, lin: Vec3) -> Self {
        Spatial { ang, lin }
    }
    #[inline]
    pub fn dot(&self, o: &Spatial) -> Real {
        self.ang.dot(o.ang) + self.lin.dot(o.lin)
    }
    #[inline]
    pub fn scale(&self, s: Real) -> Spatial {
        Spatial { ang: self.ang * s, lin: self.lin * s }
    }
    /// Motion cross motion: self x m.
    #[inline]
    pub fn cross_motion(&self, m: &Spatial) -> Spatial {
        Spatial {
            ang: self.ang.cross(m.ang),
            lin: self.ang.cross(m.lin) + self.lin.cross(m.ang),
        }
    }
    /// Motion cross force: self x* f.
    #[inline]
    pub fn cross_force(&self, f: &Spatial) -> Spatial {
        Spatial {
            ang: self.ang.cross(f.ang) + self.lin.cross(f.lin),
            lin: self.ang.cross(f.lin),
        }
    }
    /// Linear velocity of world point `p` for this motion vector.
    #[inline]
    pub fn point_velocity(&self, p: Vec3) -> Vec3 {
        self.lin + self.ang.cross(p)
    }
}

impl Add for Spatial {
    type Output = Spatial;
    #[inline]
    fn add(self, o: Spatial) -> Spatial {
        Spatial { ang: self.ang + o.ang, lin: self.lin + o.lin }
    }
}
impl AddAssign for Spatial {
    #[inline]
    fn add_assign(&mut self, o: Spatial) {
        self.ang += o.ang;
        self.lin += o.lin;
    }
}
impl Sub for Spatial {
    type Output = Spatial;
    #[inline]
    fn sub(self, o: Spatial) -> Spatial {
        Spatial { ang: self.ang - o.ang, lin: self.lin - o.lin }
    }
}

/// Rigid-body spatial inertia about the world origin, stored compactly:
/// mass, first moment m*c, and rotational inertia about the origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpatialInertia {
    pub mass: Real,
    pub mc: Vec3,
    pub inertia_o: Mat3,
}

impl Default for SpatialInertia {
    fn default() -> Self {
        SpatialInertia { mass: 0.0, mc: Vec3::ZERO, inertia_o: Mat3::ZERO }
    }
}

impl SpatialInertia {
    /// From mass, world com, and rotational inertia about com in world axes.
    pub fn from_com(mass: Real, com: Vec3, inertia_com: &Mat3) -> Self {
        let cx = Mat3::skew(com);
        // I_O = I_c + m [c]x [c]x^T
        let io = inertia_com.add(&cx.mul_mat(&cx.transpose()).scale(mass));
        SpatialInertia { mass, mc: com * mass, inertia_o: io }
    }
    #[inline]
    pub fn add(&self, o: &SpatialInertia) -> SpatialInertia {
        SpatialInertia {
            mass: self.mass + o.mass,
            mc: self.mc + o.mc,
            inertia_o: self.inertia_o.add(&o.inertia_o),
        }
    }
    /// Momentum h = I v.
    #[inline]
    pub fn mul_motion(&self, v: &Spatial) -> Spatial {
        Spatial {
            ang: self.inertia_o.mul_vec(v.ang) + self.mc.cross(v.lin),
            lin: v.lin * self.mass - self.mc.cross(v.ang),
        }
    }
}

/// In-place dense Cholesky (lower) of an n x n row-major SPD matrix.
/// Returns false if the matrix is not positive definite.
pub fn cholesky(a: &mut [Real], n: usize) -> bool {
    use crate::simd::dot;
    for j in 0..n {
        let (head, tail) = a.split_at_mut((j + 1) * n);
        let (rj, diag) = head[j * n..].split_at_mut(j);
        let rj: &[Real] = rj;
        let d = diag[0] - dot(rj, rj);
        if d <= 0.0 {
            return false;
        }
        let d = d.sqrt();
        diag[0] = d;
        let inv = 1.0 / d;
        for i in (j + 1)..n {
            let ri = &mut tail[(i - j - 1) * n..(i - j) * n];
            ri[j] = (ri[j] - dot(&ri[..j], rj)) * inv;
        }
    }
    true
}

/// Solve L L^T x = b in place given the lower factor from [`cholesky`].
pub fn cholesky_solve(l: &[Real], n: usize, x: &mut [Real]) {
    use crate::simd::{axpy, dot};
    // forward: L y = b, row-wise dot products
    for i in 0..n {
        x[i] = (x[i] - dot(&l[i * n..i * n + i], &x[..i])) / l[i * n + i];
    }
    // backward: L^T x = y, as row-wise axpy updates so memory access stays
    // contiguous (the column walk of L^T would stride by n)
    for i in (0..n).rev() {
        x[i] /= l[i * n + i];
        let xi = x[i];
        axpy(-xi, &l[i * n..i * n + i], &mut x[..i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quat_mat_roundtrip() {
        let q = Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0), 1.1);
        let q2 = Quat::from_mat(&q.to_mat());
        for i in 0..4 {
            assert!((q.0[i] - q2.0[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn cholesky_solves() {
        let mut a = vec![4.0, 2.0, 0.6, 2.0, 5.0, 1.0, 0.6, 1.0, 3.0];
        let orig = a.clone();
        assert!(cholesky(&mut a, 3));
        let mut x = vec![1.0, 2.0, 3.0];
        cholesky_solve(&a, 3, &mut x);
        for i in 0..3 {
            let r: Real = (0..3).map(|k| orig[i * 3 + k] * x[k]).sum();
            assert!((r - [1.0, 2.0, 3.0][i]).abs() < 1e-12);
        }
    }
}
