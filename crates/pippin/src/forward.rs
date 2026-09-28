//! Smooth dynamics: kinematics, composite-rigid-body mass matrix, recursive
//! Newton-Euler bias forces, passive and actuator forces, and time stepping.
//!
//! All spatial quantities live in world axes about the world origin, so
//! composite inertias and body forces accumulate by plain addition.

use crate::collision::{self, Contact, GeomPose, Hit};
use crate::data::Data;
use crate::math::{cholesky, cholesky_solve, Mat3, Quat, Real, Spatial, SpatialInertia, Vec3};
use crate::model::{JointType, Model, SolverKind};
use crate::{newton, solver};

/// Positions of bodies, joint frames, geoms, and dof motion subspaces.
pub fn kinematics(m: &Model, d: &mut Data) {
    d.xpos[0] = Vec3::ZERO;
    d.xquat[0] = Quat::IDENTITY;
    d.xmat[0] = Mat3::IDENTITY;
    d.xipos[0] = Vec3::ZERO;

    for b in 1..m.nbody() {
        let p = m.body_parent[b];
        let mut pos = d.xpos[p] + d.xmat[p].mul_vec(m.body_pos[b]);
        let mut quat = d.xquat[p].mul(m.body_quat[b]);

        for j in m.body_jntadr[b]..m.body_jntadr[b] + m.body_jntnum[b] {
            let qa = m.jnt_qposadr[j];
            let da = m.jnt_dofadr[j];
            let q = &d.qpos;
            match m.jnt_type[j] {
                JointType::Free => {
                    pos = Vec3::new(q[qa], q[qa + 1], q[qa + 2]);
                    quat = Quat([q[qa + 3], q[qa + 4], q[qa + 5], q[qa + 6]]).normalized();
                    d.xanchor[j] = pos;
                    d.xaxis[j] = Vec3::Z;
                    let r = quat.to_mat();
                    for k in 0..3 {
                        let mut e = Vec3::ZERO;
                        e[k] = 1.0;
                        d.cdof[da + k] = Spatial::new(Vec3::ZERO, e);
                        let a = r.col(k);
                        d.cdof[da + 3 + k] = Spatial::new(a, pos.cross(a));
                    }
                }
                JointType::Ball => {
                    let anchor = pos + quat.rotate(m.jnt_pos[j]);
                    let ql = Quat([q[qa], q[qa + 1], q[qa + 2], q[qa + 3]]).normalized();
                    quat = quat.mul(ql);
                    pos = anchor - quat.rotate(m.jnt_pos[j]);
                    d.xanchor[j] = anchor;
                    d.xaxis[j] = Vec3::Z;
                    let r = quat.to_mat();
                    for k in 0..3 {
                        let a = r.col(k);
                        d.cdof[da + k] = Spatial::new(a, anchor.cross(a));
                    }
                }
                JointType::Hinge => {
                    let anchor = pos + quat.rotate(m.jnt_pos[j]);
                    let axis = quat.rotate(m.jnt_axis[j]);
                    quat = quat.mul(Quat::from_axis_angle(m.jnt_axis[j], q[qa] - m.qpos0[qa]));
                    pos = anchor - quat.rotate(m.jnt_pos[j]);
                    d.xanchor[j] = anchor;
                    d.xaxis[j] = axis;
                    d.cdof[da] = Spatial::new(axis, anchor.cross(axis));
                }
                JointType::Slide => {
                    let anchor = pos + quat.rotate(m.jnt_pos[j]);
                    let axis = quat.rotate(m.jnt_axis[j]);
                    pos += axis * (q[qa] - m.qpos0[qa]);
                    d.xanchor[j] = anchor;
                    d.xaxis[j] = axis;
                    d.cdof[da] = Spatial::new(Vec3::ZERO, axis);
                }
            }
        }

        let quat = quat.normalized();
        d.xpos[b] = pos;
        d.xquat[b] = quat;
        d.xmat[b] = quat.to_mat();
        d.xipos[b] = pos + d.xmat[b].mul_vec(m.body_ipos[b]);
        d.cinert[b] = SpatialInertia::from_com(m.body_mass[b], d.xipos[b], &m.body_inertia[b].rotate(&d.xmat[b]));
    }

    for g in 0..m.ngeom() {
        let b = m.geom_body[g];
        d.geom_xpos[g] = d.xpos[b] + d.xmat[b].mul_vec(m.geom_pos[g]);
        d.geom_xmat[g] = d.xmat[b].mul_mat(&m.geom_quat[g].to_mat());
    }
}

/// Joint-space inertia via the composite rigid body algorithm.
pub fn crba(m: &Model, d: &mut Data) {
    let nv = m.nv;
    d.crb.copy_from_slice(&d.cinert);
    for b in (1..m.nbody()).rev() {
        let p = m.body_parent[b];
        d.crb[p] = d.crb[p].add(&d.crb[b]);
    }
    d.qm.fill(0.0);
    for i in 0..nv {
        let f = d.crb[m.dof_body[i]].mul_motion(&d.cdof[i]);
        let mut j = i;
        while j != usize::MAX {
            let v = d.cdof[j].dot(&f);
            d.qm[i * nv + j] = v;
            d.qm[j * nv + i] = v;
            j = m.dof_parent[j];
        }
        d.qm[i * nv + i] += m.dof_armature[i];
    }
}

/// Body velocities and the time derivative of each dof's motion subspace.
pub fn velocity(m: &Model, d: &mut Data) {
    d.cvel[0] = Spatial::ZERO;
    for b in 1..m.nbody() {
        let mut v = d.cvel[m.body_parent[b]];
        for j in m.body_jntadr[b]..m.body_jntadr[b] + m.body_jntnum[b] {
            let da = m.jnt_dofadr[j];
            match m.jnt_type[j] {
                JointType::Free | JointType::Ball => {
                    let (lin, rot) = if m.jnt_type[j] == JointType::Free { (3, da + 3) } else { (0, da) };
                    for k in 0..lin {
                        d.cdof_dot[da + k] = Spatial::ZERO;
                        v += d.cdof[da + k].scale(d.qvel[da + k]);
                    }
                    // all rotational dofs share the same velocity (matches MuJoCo)
                    for k in 0..3 {
                        d.cdof_dot[rot + k] = v.cross_motion(&d.cdof[rot + k]);
                    }
                    for k in 0..3 {
                        v += d.cdof[rot + k].scale(d.qvel[rot + k]);
                    }
                }
                _ => {
                    d.cdof_dot[da] = v.cross_motion(&d.cdof[da]);
                    v += d.cdof[da].scale(d.qvel[da]);
                }
            }
        }
        d.cvel[b] = v;
    }
}

/// Coriolis, centrifugal, and gravity forces via recursive Newton-Euler.
pub fn rne(m: &Model, d: &mut Data) {
    let nb = m.nbody();
    let s = &mut d.scratch;
    s.cacc.resize(nb, Spatial::ZERO);
    s.cfrc.resize(nb, Spatial::ZERO);
    s.cacc[0] = Spatial::new(Vec3::ZERO, -m.gravity);
    s.cfrc[0] = Spatial::ZERO;
    for b in 1..nb {
        let mut a = s.cacc[m.body_parent[b]];
        for k in m.body_dofadr[b]..m.body_dofadr[b] + m.body_dofnum[b] {
            a += d.cdof_dot[k].scale(d.qvel[k]);
        }
        s.cacc[b] = a;
        let h = d.cinert[b].mul_motion(&d.cvel[b]);
        s.cfrc[b] = d.cinert[b].mul_motion(&a) + d.cvel[b].cross_force(&h);
    }
    for b in (1..nb).rev() {
        let p = m.body_parent[b];
        let f = s.cfrc[b];
        s.cfrc[p] += f;
    }
    for k in 0..m.nv {
        d.qfrc_bias[k] = d.cdof[k].dot(&s.cfrc[m.dof_body[k]]);
    }
}

pub fn passive(m: &Model, d: &mut Data) {
    for k in 0..m.nv {
        d.qfrc_passive[k] = -m.dof_damping[k] * d.qvel[k];
    }
    for j in 0..m.njnt() {
        let kp = m.jnt_stiffness[j];
        if kp != 0.0 && matches!(m.jnt_type[j], JointType::Hinge | JointType::Slide) {
            d.qfrc_passive[m.jnt_dofadr[j]] -= kp * (d.qpos[m.jnt_qposadr[j]] - m.jnt_springref[j]);
        }
    }
}

pub fn actuation(m: &Model, d: &mut Data) {
    d.qfrc_actuator.fill(0.0);
    for u in 0..m.nu {
        let j = m.actuator_joint[u];
        let gear = m.actuator_gear[u];
        let length = gear * d.qpos[m.jnt_qposadr[j]];
        let vel = gear * d.qvel[m.jnt_dofadr[j]];
        let mut ctrl = d.ctrl[u];
        if m.actuator_ctrllimited[u] {
            let r = m.actuator_ctrlrange[u];
            ctrl = ctrl.clamp(r[0], r[1]);
        }
        let b = m.actuator_bias[u];
        let mut f = m.actuator_gain[u] * ctrl + b[0] + b[1] * length + b[2] * vel;
        if m.actuator_forcelimited[u] {
            let r = m.actuator_forcerange[u];
            f = f.clamp(r[0], r[1]);
        }
        d.actuator_force[u] = f;
        d.qfrc_actuator[m.jnt_dofadr[j]] += gear * f;
    }
}

/// Broad + narrow phase.
pub fn collide(m: &Model, d: &mut Data) {
    d.contacts.clear();
    let margin = m.solver.contact_margin;
    let mut hits: Vec<Hit> = std::mem::take(&mut d.scratch.hits);
    for &(g1, g2) in &m.collision_pairs {
        // bounding-sphere cull (planes have infinite radius)
        let r = m.geom_rbound[g1] + m.geom_rbound[g2] + margin;
        if r.is_finite() && (d.geom_xpos[g1] - d.geom_xpos[g2]).norm2() > r * r {
            continue;
        }
        let pose = |g: usize| GeomPose { typ: m.geom_type[g], pos: d.geom_xpos[g], mat: d.geom_xmat[g], size: m.geom_size[g] };
        hits.clear();
        collision::collide(&pose(g1), &pose(g2), margin, &mut hits);
        for (i, h) in hits.iter().enumerate() {
            d.contacts.push(Contact {
                pos: h.pos,
                normal: h.normal,
                tangent: h.tangent,
                depth: h.depth,
                geom: [g1, g2],
                feature: i as u32,
            });
        }
    }
    d.scratch.hits = hits;
}

/// Compute all position- and velocity-dependent quantities (no integration).
pub fn forward(m: &Model, d: &mut Data) {
    kinematics(m, d);
    crba(m, d);
    velocity(m, d);
    rne(m, d);
    passive(m, d);
    actuation(m, d);
    collide(m, d);
}

/// Compute model constants that depend on the reference configuration:
/// per-body and per-dof inverse inertia (MuJoCo's `invweight0`).
pub fn set_const(m: &mut Model) {
    let mut d = Data::new(m);
    kinematics(m, &mut d);
    crba(m, &mut d);
    let nv = m.nv;
    m.dof_invweight = vec![0.0; nv];
    m.body_invweight = vec![[0.0; 2]; m.nbody()];
    if nv == 0 {
        return;
    }
    let mut l = d.qm.clone();
    if !cholesky(&mut l, nv) {
        return;
    }
    let quad = |row: &[Real], x: &mut [Real]| -> Real {
        x.copy_from_slice(row);
        cholesky_solve(&l, nv, x);
        row.iter().zip(x.iter()).map(|(a, b)| a * b).sum()
    };
    let mut x = vec![0.0; nv];
    let mut e = vec![0.0; nv];
    for k in 0..nv {
        e.fill(0.0);
        e[k] = 1.0;
        m.dof_invweight[k] = quad(&e, &mut x);
    }
    // multi-dof joints share one averaged value per translational/rotational group
    for j in 0..m.njnt() {
        let da = m.jnt_dofadr[j];
        let groups: &[usize] = match m.jnt_type[j] {
            JointType::Free => &[0, 3],
            JointType::Ball => &[0],
            _ => continue,
        };
        for &g in groups {
            let avg = (0..3).map(|i| m.dof_invweight[da + g + i]).sum::<Real>() / 3.0;
            (0..3).for_each(|i| m.dof_invweight[da + g + i] = avg);
        }
    }
    let mut row = vec![0.0; nv];
    for b in 1..m.nbody() {
        let mut w = [0.0; 2];
        for axis in 0..3 {
            let mut dir = Vec3::ZERO;
            dir[axis] = 1.0;
            for (kind, slot) in w.iter_mut().enumerate() {
                row.fill(0.0);
                let mut k = m.body_lastdof[b];
                while k != usize::MAX {
                    let c = &d.cdof[k];
                    row[k] = if kind == 0 { dir.dot(c.point_velocity(d.xipos[b])) } else { dir.dot(c.ang) };
                    k = m.dof_parent[k];
                }
                *slot += quad(&row, &mut x) / 3.0;
            }
        }
        m.body_invweight[b] = w;
    }
}

/// Advance the simulation by one timestep.
pub fn step(m: &Model, d: &mut Data) {
    forward(m, d);
    let nv = m.nv;
    let h = m.timestep;

    let damped = m.dof_damping.iter().any(|&b| b != 0.0);
    let mut smooth = std::mem::take(&mut d.scratch.smooth);
    smooth.clear();
    smooth.extend((0..nv).map(|k| d.qfrc_passive[k] + d.qfrc_actuator[k] + d.qfrc_applied[k] - d.qfrc_bias[k]));
    let mut acc = std::mem::take(&mut d.scratch.tmp_nv);
    acc.clear();
    acc.extend_from_slice(&smooth);

    let mut v = std::mem::take(&mut d.scratch.v);
    let mut v_pos = std::mem::take(&mut d.scratch.v_pos);
    v.clear();
    match m.solver.kind {
        SolverKind::Newton => {
            // As in MuJoCo: solve constraints with M, then apply implicit
            // damping: qacc = (M + hD)^-1 (qfrc_smooth + qfrc_constraint).
            factor_mass(m, d, 0.0);
            cholesky_solve(&d.qm_chol, nv, &mut acc);
            newton::solve(m, d, &mut acc);
            if damped {
                factor_mass(m, d, h);
                for k in 0..nv {
                    acc[k] = smooth[k] + d.qfrc_constraint[k];
                }
                cholesky_solve(&d.qm_chol, nv, &mut acc);
            }
            v.extend((0..nv).map(|k| d.qvel[k] + h * acc[k]));
            v_pos.clear();
            v_pos.extend_from_slice(&v);
        }
        SolverKind::Pgs => {
            // PGS works on velocities with the damped inertia throughout
            factor_mass(m, d, h);
            cholesky_solve(&d.qm_chol, nv, &mut acc);
            v.extend((0..nv).map(|k| d.qvel[k] + h * acc[k]));
            solver::solve(m, d, &mut v, &mut v_pos);
        }
    }
    d.scratch.tmp_nv = acc;
    d.scratch.smooth = smooth;

    for k in 0..nv {
        d.qacc[k] = (v[k] - d.qvel[k]) / h;
    }
    integrate_pos(m, &mut d.qpos, &v_pos, h);
    d.qvel.copy_from_slice(&v);
    d.scratch.v = v;
    d.scratch.v_pos = v_pos;
    d.time += h;
}

/// Factor M + h*D (damping-implicit inertia) into `d.qm_chol`.
fn factor_mass(m: &Model, d: &mut Data, h: Real) {
    let nv = m.nv;
    d.qm_chol.copy_from_slice(&d.qm);
    for k in 0..nv {
        d.qm_chol[k * nv + k] += h * m.dof_damping[k];
    }
    if !cholesky(&mut d.qm_chol, nv) {
        panic!("mass matrix is not positive definite");
    }
}

/// qpos <- qpos (+) h * qvel on the configuration manifold.
pub fn integrate_pos(m: &Model, qpos: &mut [Real], qvel: &[Real], h: Real) {
    for j in 0..m.njnt() {
        let qa = m.jnt_qposadr[j];
        let da = m.jnt_dofadr[j];
        match m.jnt_type[j] {
            JointType::Free => {
                for k in 0..3 {
                    qpos[qa + k] += h * qvel[da + k];
                }
                let q = Quat([qpos[qa + 3], qpos[qa + 4], qpos[qa + 5], qpos[qa + 6]]);
                let w = Vec3::new(qvel[da + 3], qvel[da + 4], qvel[da + 5]);
                qpos[qa + 3..qa + 7].copy_from_slice(&q.integrate_local(w, h).0);
            }
            JointType::Ball => {
                let q = Quat([qpos[qa], qpos[qa + 1], qpos[qa + 2], qpos[qa + 3]]);
                let w = Vec3::new(qvel[da], qvel[da + 1], qvel[da + 2]);
                qpos[qa..qa + 4].copy_from_slice(&q.integrate_local(w, h).0);
            }
            _ => qpos[qa] += h * qvel[da],
        }
    }
}
