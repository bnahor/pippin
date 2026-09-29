//! Convex primal Newton constraint solver (MuJoCo-style).
//!
//! Finds the constrained acceleration a minimizing
//!
//!   f(a) = 1/2 (a - a0)' M (a - a0) + sum_i s_i(J_i a - aref_i)
//!
//! where a0 is the unconstrained acceleration, M the (implicitly damped)
//! joint-space inertia, and s_i(r) = 1/2 D_i r^2 when r < 0, else 0. Every row
//! is a one-sided inequality: friction uses pyramidal cones (rows J_n +/- mu J_t),
//! so normal force and friction are coupled only through the shared cost.
//! f is convex and piecewise quadratic, so Newton with an exact line search
//! typically converges in a handful of iterations.

use crate::data::Data;
use crate::math::{cholesky, cholesky_solve, Real};
use crate::simd;
use crate::model::{JointType, Model};
use crate::solver::point_jac;

/// Per-row constants of the soft constraint model.
#[derive(Clone, Copy, Debug, Default)]
pub struct NRow {
    pub aref: Real,
    /// Constraint stiffness weight D = 1 / R.
    pub d: Real,
    /// Equality (two-sided, always active) rather than inequality.
    pub bilateral: bool,
}

impl NRow {
    /// Whether the row contributes cost at residual `r = J a - aref`.
    #[inline]
    pub fn active(&self, r: Real) -> bool {
        self.bilateral || r < 0.0
    }
}

/// MuJoCo impedance curve: how "hard" a constraint is at violation `pos`.
fn impedance(solimp: &[Real; 5], pos: Real) -> Real {
    let [dmin, dmax, width, mid, power] = *solimp;
    let x = (pos.abs() / width.max(1e-12)).min(1.0);
    if x >= 1.0 || dmin == dmax {
        return dmax;
    }
    // power 2 is the default; avoid libm pow() on that hot path
    let pw = |b: Real, e: Real| if e == 2.0 { b * b } else if e == 1.0 { b } else { b.powf(e) };
    let y = if x <= mid {
        pw(x, power) / pw(mid, power - 1.0)
    } else {
        1.0 - pw(1.0 - x, power) / pw(1.0 - mid, power - 1.0)
    };
    (dmin + y * (dmax - dmin)).clamp(1e-4, 0.9999)
}

struct Workspace<'a> {
    nv: usize,
    nr: usize,
    jac: &'a [Real],
    rows: &'a [NRow],
    qm: &'a [Real],
    damp_h: &'a [Real],
}

impl Workspace<'_> {
    /// y = M_hat x
    fn mul_m(&self, x: &[Real], y: &mut [Real]) {
        let nv = self.nv;
        for i in 0..nv {
            let row = &self.qm[i * nv..(i + 1) * nv];
            y[i] = simd::dot(row, x) + self.damp_h[i] * x[i];
        }
    }

    fn jrow(&self, r: usize) -> &[Real] {
        &self.jac[r * self.nv..(r + 1) * self.nv]
    }

    fn dot_j(&self, r: usize, x: &[Real]) -> Real {
        simd::dot(self.jrow(r), x)
    }

    /// Total cost at `a`; also fills `jar` = J a - aref.
    fn cost(&self, a: &[Real], a0: &[Real], jar: &mut [Real], tmp: &mut [Real], tmp2: &mut [Real]) -> Real {
        for k in 0..self.nv {
            tmp[k] = a[k] - a0[k];
        }
        self.mul_m(tmp, tmp2);
        let mut c = 0.5 * tmp.iter().zip(tmp2.iter()).map(|(x, y)| x * y).sum::<Real>();
        for r in 0..self.nr {
            jar[r] = self.dot_j(r, a) - self.rows[r].aref;
            if self.rows[r].active(jar[r]) {
                c += 0.5 * self.rows[r].d * jar[r] * jar[r];
            }
        }
        c
    }
}

/// Build Jacobian rows and soft-constraint constants. Returns row count.
fn build_rows(m: &Model, d: &mut Data) -> usize {
    let nv = m.nv;
    let h = m.timestep;
    let opt = &m.solver;
    let jac = &mut d.scratch.jac;
    let rows = &mut d.scratch.nrows;
    let pos = &mut d.scratch.lambda; // reused: per-row violation
    let diag = &mut d.scratch.v_pos; // reused: per-row MuJoCo diagApprox
    jac.clear();
    rows.clear();
    pos.clear();
    diag.clear();

    let mut jn = vec![0.0; nv];
    let mut jt1 = vec![0.0; nv];
    let mut jt2 = vec![0.0; nv];
    for c in d.contacts.iter() {
        if c.depth <= 0.0 {
            continue; // within margin but not touching
        }
        let (g1, g2) = (c.geom[0], c.geom[1]);
        let (b1, b2) = (m.geom_body[g1], m.geom_body[g2]);
        let mu = m.geom_friction[g1].max(m.geom_friction[g2]);
        let (t1, t2) = c.frame();
        for (row, dir) in [(&mut jn, c.normal), (&mut jt1, t1), (&mut jt2, t2)] {
            row.fill(0.0);
            point_jac(m, &d.cdof, b2, c.pos, dir, 1.0, row);
            point_jac(m, &d.cdof, b1, c.pos, dir, -1.0, row);
        }
        // MuJoCo's pyramidal regularization: R = 2 mu^2 (1 + mu^2) tran (1-imp)/imp / impratio
        let tran = m.body_invweight[b1][0] + m.body_invweight[b2][0];
        let dpyr = 2.0 * mu * mu * (1.0 + mu * mu) * tran / opt.impratio;
        for t in [&jt1, &jt2] {
            for s in [1.0, -1.0] {
                jac.extend((0..nv).map(|k| jn[k] + s * mu * t[k]));
                pos.push(-c.depth);
                diag.push(dpyr);
            }
        }
    }
    for j in 0..m.njnt() {
        if !m.jnt_limited[j] || !matches!(m.jnt_type[j], JointType::Hinge | JointType::Slide) {
            continue;
        }
        let q = d.qpos[m.jnt_qposadr[j]];
        let da = m.jnt_dofadr[j];
        let [lo, hi] = m.jnt_range[j];
        for (sep, sign) in [(q - lo, 1.0), (hi - q, -1.0)] {
            if sep < 0.0 {
                let start = jac.len();
                jac.resize(start + nv, 0.0);
                jac[start + da] = sign;
                pos.push(sep);
                diag.push(m.dof_invweight[da]);
            }
        }
    }

    // joint equality: (q1 - q1_0) - poly(q2 - q2_0) = 0, with its own softness
    let n_ineq = pos.len();
    let mut eq_params: Vec<([Real; 2], [Real; 5])> = vec![];
    for e in 0..m.eq_joint1.len() {
        let (j1, j2) = (m.eq_joint1[e], m.eq_joint2[e]);
        let c = m.eq_polycoef[e];
        let q1 = d.qpos[m.jnt_qposadr[j1]] - m.qpos0[m.jnt_qposadr[j1]];
        let (mut poly, mut dpoly) = (c[0], 0.0);
        if j2 != usize::MAX {
            let x = d.qpos[m.jnt_qposadr[j2]] - m.qpos0[m.jnt_qposadr[j2]];
            poly = c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * c[4])));
            dpoly = c[1] + x * (2.0 * c[2] + x * (3.0 * c[3] + x * 4.0 * c[4]));
        }
        let start = jac.len();
        jac.resize(start + nv, 0.0);
        jac[start + m.jnt_dofadr[j1]] += 1.0;
        let mut dg = m.dof_invweight[m.jnt_dofadr[j1]];
        if j2 != usize::MAX {
            jac[start + m.jnt_dofadr[j2]] -= dpoly;
            dg += m.dof_invweight[m.jnt_dofadr[j2]];
        }
        pos.push(q1 - poly);
        diag.push(dg);
        eq_params.push((m.eq_solref[e], m.eq_solimp[e]));
    }

    let nr = pos.len();
    if nr == 0 {
        return 0;
    }

    // reference acceleration and regularization per row
    for r in 0..nr {
        let bilateral = r >= n_ineq;
        let (solref, solimp) = if bilateral { eq_params[r - n_ineq] } else { (opt.solref, opt.solimp) };
        let [tc, dr] = solref;
        let tc = tc.max(2.0 * h);
        let dmax = solimp[1];
        let b = 2.0 / (dmax * tc);
        let k = 1.0 / (dmax * dmax * tc * tc * dr * dr);
        let jr = &jac[r * nv..(r + 1) * nv];
        let vel: Real = jr.iter().zip(&d.qvel).map(|(a, b)| a * b).sum();
        let imp = impedance(&solimp, pos[r]);
        let reg = ((1.0 - imp) / imp * diag[r]).max(1e-15);
        rows.push(NRow { aref: -b * vel - k * imp * pos[r], d: 1.0 / reg, bilateral });
    }
    nr
}

/// Replace the unconstrained acceleration `a` with the constrained one.
pub fn solve(m: &Model, d: &mut Data, a: &mut [Real]) {
    let nv = m.nv;
    let nr = build_rows(m, d);
    d.qfrc_constraint.fill(0.0);
    if nr == 0 {
        return;
    }
    let s = &mut d.scratch;
    // constraints are solved against the undamped inertia (see forward::step)
    s.damp_h.clear();
    s.damp_h.resize(nv, 0.0);
    let mut buf = std::mem::take(&mut s.newton_buf);
    buf.clear();
    buf.resize(6 * nv + 2 * nr + nv * nv, 0.0);
    let (a0, rest) = buf.split_at_mut(nv);
    let (grad, rest) = rest.split_at_mut(nv);
    let (p, rest) = rest.split_at_mut(nv);
    let (tmp, rest) = rest.split_at_mut(nv);
    let (tmp2, rest) = rest.split_at_mut(nv);
    let (mp, rest) = rest.split_at_mut(nv);
    let (jar, rest) = rest.split_at_mut(nr);
    let (jp, hess) = rest.split_at_mut(nr);
    a0.copy_from_slice(a);

    let w = Workspace { nv, nr, jac: &s.jac, rows: &s.nrows, qm: &d.qm, damp_h: &s.damp_h };

    // warm start from the previous solution when it is better
    if d.qacc.len() == nv {
        let c0 = w.cost(a0, a0, jar, tmp, tmp2);
        let c1 = w.cost(&d.qacc, a0, jar, tmp, tmp2);
        if c1 < c0 {
            a.copy_from_slice(&d.qacc);
        }
    }

    // gradient scale for the convergence test
    w.mul_m(a0, tmp);
    let scale = 1.0 + tmp.iter().map(|x| x * x).sum::<Real>().sqrt();

    let mut iters = 0;
    let mut prev_active: u128 = u128::MAX;
    while iters < m.solver.iterations.max(1) {
        iters += 1;
        // residuals, active set, gradient
        let mut active: u128 = 0;
        for k in 0..nv {
            tmp[k] = a[k] - a0[k];
        }
        w.mul_m(tmp, grad);
        for r in 0..nr {
            jar[r] = w.dot_j(r, a) - w.rows[r].aref;
            if w.rows[r].active(jar[r]) {
                if r < 128 {
                    active |= 1 << r;
                }
                let f = w.rows[r].d * jar[r];
                for (g, j) in grad.iter_mut().zip(w.jrow(r)) {
                    *g += f * j;
                }
            }
        }
        let gnorm = grad.iter().map(|x| x * x).sum::<Real>().sqrt();
        if gnorm < m.solver.tolerance * scale || (nr <= 128 && active == prev_active && iters > 1) {
            break;
        }
        prev_active = active;

        // Hessian H = M_hat + J_A' D J_A, Newton direction p = -H^-1 g
        for i in 0..nv {
            for j in 0..nv {
                hess[i * nv + j] = w.qm[i * nv + j];
            }
            hess[i * nv + i] += w.damp_h[i];
        }
        for r in 0..nr {
            if w.rows[r].active(jar[r]) {
                let dr = w.rows[r].d;
                let jr = w.jrow(r);
                for i in 0..nv {
                    if jr[i] == 0.0 {
                        continue;
                    }
                    let di = dr * jr[i];
                    for j in 0..nv {
                        hess[i * nv + j] += di * jr[j];
                    }
                }
            }
        }
        if !cholesky(hess, nv) {
            break;
        }
        for k in 0..nv {
            p[k] = -grad[k];
        }
        cholesky_solve(hess, nv, p);

        // exact line search on the piecewise-quadratic cost along p
        w.mul_m(p, mp);
        let mpp: Real = p.iter().zip(mp.iter()).map(|(x, y)| x * y).sum();
        let mpg: Real = tmp.iter().zip(mp.iter()).map(|(x, y)| x * y).sum();
        for r in 0..nr {
            jp[r] = w.dot_j(r, p);
        }
        let deriv = |al: Real| -> (Real, Real) {
            let (mut d1, mut d2) = (al * mpp + mpg, mpp);
            for r in 0..nr {
                let v = jar[r] + al * jp[r];
                if w.rows[r].active(v) {
                    let dq = w.rows[r].d * jp[r];
                    d1 += dq * v;
                    d2 += dq * jp[r];
                }
            }
            (d1, d2)
        };
        let (mut lo, mut hi) = (0.0, Real::INFINITY);
        let mut al = 1.0;
        let (d1_0, _) = deriv(0.0);
        let tol = 1e-12 * d1_0.abs().max(1e-30);
        for _ in 0..50 {
            let (d1, d2) = deriv(al);
            if d1.abs() <= tol {
                break;
            }
            if d1 < 0.0 {
                lo = al;
            } else {
                hi = al;
            }
            let mut next = al - d1 / d2.max(1e-30);
            if next <= lo || next >= hi {
                next = if hi.is_finite() { 0.5 * (lo + hi) } else { 2.0 * al.max(lo) };
            }
            if (next - al).abs() <= 1e-14 * al.abs().max(1.0) {
                al = next;
                break;
            }
            al = next;
        }
        for k in 0..nv {
            a[k] += al * p[k];
        }
    }

    // constraint force at the solution: qfrc_c = -J' D (J a - aref)_-
    for r in 0..nr {
        let v = w.dot_j(r, a) - w.rows[r].aref;
        if w.rows[r].active(v) {
            let f = -w.rows[r].d * v;
            for (q, j) in d.qfrc_constraint.iter_mut().zip(w.jrow(r)) {
                *q += f * j;
            }
        }
    }
    s.newton_iters = iters;
    s.newton_buf = buf;
}

#[cfg(test)]
mod tests {
    use super::impedance;

    #[test]
    fn impedance_curve() {
        let s = [0.9, 0.95, 0.001, 0.5, 2.0];
        assert!((impedance(&s, 0.0) - 0.9).abs() < 1e-12);
        assert!((impedance(&s, 0.0005) - 0.925).abs() < 1e-12);
        assert!((impedance(&s, 0.01) - 0.95).abs() < 1e-12);
    }
}
