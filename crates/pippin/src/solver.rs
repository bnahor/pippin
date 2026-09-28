//! Velocity-level constraint solver: soft projected Gauss-Seidel in joint space.
//!
//! Each constraint row j has Jacobian J_j (1 x nv). With M_hat the (damped)
//! joint-space inertia, the impulse lambda changes velocities by
//! M_hat^-1 J^T lambda. Contacts use a mass-independent soft spring (as in
//! Box2D v3's soft step) so stiffness is stable at any timestep. Positions are
//! integrated with the biased velocity; the stored velocity is re-solved without
//! bias ("relax"), which removes the energy the position correction would add.

use std::f64::consts::PI;

use crate::data::Data;
use crate::math::{cholesky_solve, Real, Spatial, Vec3};
use crate::model::{JointType, Model};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RowKind {
    /// Unilateral: lambda >= 0. `sep` is the signed separation.
    Normal { sep: Real },
    /// Box-bounded by mu * lambda of row `normal`.
    Friction { normal: usize, mu: Real },
}

#[derive(Clone, Copy, Debug)]
pub struct Row {
    pub kind: RowKind,
    pub inv_a: Real,
    /// Contact index (for warm start bookkeeping) or usize::MAX for limits.
    pub contact: usize,
}

const MAX_BIAS_VELOCITY: Real = 1.0;
/// Stop once no row changes its constraint velocity by more than this (m/s).
const TOLERANCE: Real = 1e-5;

struct Soft {
    bias_rate: Real,
    mass_scale: Real,
    impulse_scale: Real,
}

fn soft(hertz: Real, zeta: Real, h: Real) -> Soft {
    let omega = 2.0 * PI * hertz;
    let a1 = 2.0 * zeta + h * omega;
    let a2 = h * omega * a1;
    let a3 = 1.0 / (1.0 + a2);
    Soft { bias_rate: omega / a1, mass_scale: a2 * a3, impulse_scale: a3 }
}

/// Accumulate the Jacobian of a world point on `body` along `dir` into `row`.
pub(crate) fn point_jac(m: &Model, cdof: &[Spatial], body: usize, p: Vec3, dir: Vec3, sign: Real, row: &mut [Real]) {
    let mut k = m.body_lastdof[body];
    while k != usize::MAX {
        row[k] += sign * dir.dot(cdof[k].point_velocity(p));
        k = m.dof_parent[k];
    }
}

/// Solve constraints in place on `v` (the unconstrained next velocity).
/// Writes the velocity to integrate positions with into `v_pos`.
pub fn solve(m: &Model, d: &mut Data, v: &mut [Real], v_pos: &mut Vec<Real>) {
    let nv = m.nv;
    let h = m.timestep;
    let opt = &m.solver;
    let mut rows = std::mem::take(&mut d.scratch.rows);
    let mut jac = std::mem::take(&mut d.scratch.jac);
    let mut mj = std::mem::take(&mut d.scratch.minv_jt);
    let mut lambda = std::mem::take(&mut d.scratch.lambda);
    rows.clear();
    jac.clear();

    // ---- build rows ----
    let push_row = |jac: &mut Vec<Real>| -> usize {
        jac.resize(jac.len() + nv, 0.0);
        jac.len() / nv - 1
    };
    for (ci, c) in d.contacts.iter().enumerate() {
        let (g1, g2) = (c.geom[0], c.geom[1]);
        let (b1, b2) = (m.geom_body[g1], m.geom_body[g2]);
        let mu = m.geom_friction[g1].max(m.geom_friction[g2]);
        let (t1, t2) = c.frame();
        let base = rows.len();
        for (r, dir) in [c.normal, t1, t2].into_iter().enumerate() {
            let idx = push_row(&mut jac);
            let row = &mut jac[idx * nv..(idx + 1) * nv];
            point_jac(m, &d.cdof, b2, c.pos, dir, 1.0, row);
            point_jac(m, &d.cdof, b1, c.pos, dir, -1.0, row);
            let kind = if r == 0 { RowKind::Normal { sep: -c.depth } } else { RowKind::Friction { normal: base, mu } };
            rows.push(Row { kind, inv_a: 0.0, contact: ci });
        }
    }
    for j in 0..m.njnt() {
        if !m.jnt_limited[j] || !matches!(m.jnt_type[j], JointType::Hinge | JointType::Slide) {
            continue;
        }
        let q = d.qpos[m.jnt_qposadr[j]];
        let da = m.jnt_dofadr[j];
        let [lo, hi] = m.jnt_range[j];
        let reach = opt.contact_margin + 2.0 * h * v[da].abs();
        for (sep, sign) in [(q - lo, 1.0), (hi - q, -1.0)] {
            if sep < reach {
                let idx = push_row(&mut jac);
                jac[idx * nv + da] = sign;
                rows.push(Row { kind: RowKind::Normal { sep }, inv_a: 0.0, contact: usize::MAX });
            }
        }
    }

    let nr = rows.len();
    v_pos.clear();
    v_pos.extend_from_slice(v);
    if nr == 0 {
        d.qfrc_constraint.fill(0.0);
        d.scratch.prev_contacts.clear();
        d.scratch.prev_lambda.clear();
        d.scratch.rows = rows;
        d.scratch.jac = jac;
        d.scratch.minv_jt = mj;
        d.scratch.lambda = lambda;
        return;
    }

    // ---- M_hat^-1 J^T and diagonal effective mass ----
    mj.clear();
    mj.extend_from_slice(&jac);
    for r in 0..nr {
        let x = &mut mj[r * nv..(r + 1) * nv];
        cholesky_solve(&d.qm_chol, nv, x);
        let a: Real = (0..nv).map(|k| jac[r * nv + k] * x[k]).sum();
        rows[r].inv_a = if a > 1e-12 { 1.0 / a } else { 0.0 };
    }

    // ---- warm start from matching contacts of the previous step ----
    lambda.clear();
    lambda.resize(nr, 0.0);
    warm_start(d, &rows, &mut lambda);
    for r in 0..nr {
        if lambda[r] != 0.0 {
            for k in 0..nv {
                v[k] += mj[r * nv + k] * lambda[r];
            }
        }
    }

    // ---- PGS ----
    let sc = soft(opt.contact_hertz, opt.contact_damping_ratio, h);
    // One Gauss-Seidel sweep; returns the largest constraint-space velocity change.
    let iterate = |v: &mut [Real], lambda: &mut [Real], use_bias: bool| -> Real {
        let mut max_dv: Real = 0.0;
        for r in 0..nr {
            let row = rows[r];
            if row.inv_a == 0.0 {
                continue;
            }
            let jr = &jac[r * nv..(r + 1) * nv];
            let jv: Real = (0..nv).map(|k| jr[k] * v[k]).sum();
            let new = match row.kind {
                RowKind::Normal { sep } => {
                    let s = sep + opt.contact_slop;
                    let (bias, ms, is) = if s > 0.0 {
                        (s / h, 1.0, 0.0)
                    } else if use_bias {
                        ((sc.bias_rate * s).max(-MAX_BIAS_VELOCITY), sc.mass_scale, sc.impulse_scale)
                    } else {
                        (0.0, 1.0, 0.0)
                    };
                    let delta = -row.inv_a * ms * (jv + bias) - is * lambda[r];
                    (lambda[r] + delta).max(0.0)
                }
                RowKind::Friction { normal, mu } => {
                    let bound = mu * lambda[normal];
                    (lambda[r] - row.inv_a * jv).clamp(-bound, bound)
                }
            };
            let dl = new - lambda[r];
            if dl != 0.0 {
                max_dv = max_dv.max(dl.abs() / row.inv_a);
                let mr = &mj[r * nv..(r + 1) * nv];
                for k in 0..nv {
                    v[k] += mr[k] * dl;
                }
                lambda[r] = new;
            }
        }
        max_dv
    };
    for _ in 0..opt.iterations {
        if iterate(v, &mut lambda, true) < TOLERANCE {
            break;
        }
    }
    v_pos.copy_from_slice(v);
    for _ in 0..(opt.iterations / 3).max(2) {
        if iterate(v, &mut lambda, false) < TOLERANCE {
            break;
        }
    }

    // ---- outputs ----
    d.qfrc_constraint.fill(0.0);
    for r in 0..nr {
        for k in 0..nv {
            d.qfrc_constraint[k] += jac[r * nv + k] * lambda[r] / h;
        }
    }
    d.scratch.prev_contacts.clear();
    d.scratch.prev_contacts.extend_from_slice(&d.contacts);
    d.scratch.prev_lambda.clear();
    d.scratch.prev_lambda.resize(d.contacts.len(), [0.0; 3]);
    for r in 0..nr {
        let c = rows[r].contact;
        if c != usize::MAX {
            let slot = match rows[r].kind {
                RowKind::Normal { .. } => 0,
                RowKind::Friction { normal, .. } => r - normal,
            };
            d.scratch.prev_lambda[c][slot] = lambda[r];
        }
    }

    d.scratch.rows = rows;
    d.scratch.jac = jac;
    d.scratch.minv_jt = mj;
    d.scratch.lambda = lambda;
}

/// Match each new contact to the nearest previous contact of the same geom pair.
fn warm_start(d: &Data, rows: &[Row], lambda: &mut [Real]) {
    const MATCH_DIST2: Real = 0.01 * 0.01;
    let prev = &d.scratch.prev_contacts;
    if prev.is_empty() {
        return;
    }
    let mut r = 0;
    while r < rows.len() {
        let ci = rows[r].contact;
        if ci == usize::MAX {
            r += 1;
            continue;
        }
        let c = &d.contacts[ci];
        let mut best = (MATCH_DIST2, usize::MAX);
        for (pi, p) in prev.iter().enumerate() {
            if p.geom == c.geom {
                let d2 = (p.pos - c.pos).norm2();
                if d2 < best.0 {
                    best = (d2, pi);
                }
            }
        }
        if best.1 != usize::MAX {
            let p = &prev[best.1];
            let l = d.scratch.prev_lambda[best.1];
            let (pt1, pt2) = p.frame();
            let (t1, t2) = c.frame();
            let f = pt1 * l[1] + pt2 * l[2];
            lambda[r] = l[0];
            lambda[r + 1] = f.dot(t1);
            lambda[r + 2] = f.dot(t2);
        }
        r += 3;
    }
}
