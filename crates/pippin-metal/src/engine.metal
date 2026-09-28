// Pippin GPU engine: one thread simulates one environment.
//
// This file mirrors the CPU reference engine (crates/pippin/src) routine for
// routine, in f32. Model sizes and constants are generated per model and
// spliced in at PIPPIN_MODEL, so every loop has compile-time bounds.

#include <metal_stdlib>
using namespace metal;

// PIPPIN_MODEL

#ifndef ABLATE
#define ABLATE 0
#endif
#define STAGE(k, expr) if (ABLATE == k) { v[0] += 1e-30f * (expr); return; }

#define JNT_FREE 0
#define JNT_BALL 1
#define JNT_SLIDE 2
#define JNT_HINGE 3

#define GEOM_PLANE 0
#define GEOM_SPHERE 1
#define GEOM_CAPSULE 2
#define GEOM_BOX 3
#define GEOM_CYLINDER 4

// ---------------------------------------------------------------- math

inline float3x3 mat_identity() { return float3x3(float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1)); }

inline float3x3 outer3(float3 a, float3 b) { return float3x3(a * b.x, a * b.y, a * b.z); }

// quaternions are float4(w, x, y, z)
inline float4 qmul(float4 a, float4 b) {
    return float4(a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                  a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                  a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
                  a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]);
}

inline float4 qnormalize(float4 q) {
    float n = length(q);
    return n < 1e-20f ? float4(1, 0, 0, 0) : q / n;
}

inline float4 qaxisangle(float3 axis, float angle) {
    float3 a = normalize(axis);
    float s = sin(0.5f * angle), c = cos(0.5f * angle);
    return float4(c, a * s);
}

inline float3x3 qmat(float4 q) {
    float w = q[0], x = q[1], y = q[2], z = q[3];
    float ww = w * w, xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, xz = x * z, yz = y * z, wx = w * x, wy = w * y, wz = w * z;
    // columns
    return float3x3(float3(ww + xx - yy - zz, 2 * (xy + wz), 2 * (xz - wy)),
                    float3(2 * (xy - wz), ww - xx + yy - zz, 2 * (yz + wx)),
                    float3(2 * (xz + wy), 2 * (yz - wx), ww - xx - yy + zz));
}

inline float4 qintegrate_local(float4 q, float3 w, float h) {
    float n = length(w);
    if (n * h < 1e-12f) return q;
    return qnormalize(qmul(q, qaxisangle(w / n, n * h)));
}

inline void tangents(float3 n, thread float3& t1, thread float3& t2) {
    float3 a = fabs(n.x) < 0.57f ? float3(1, 0, 0) : float3(0, 1, 0);
    t1 = normalize(cross(n, a));
    t2 = cross(n, t1);
}

// spatial motion/force [angular; linear] about the world origin
struct Sp {
    float3 a;
    float3 l;
};
inline Sp sp(float3 a, float3 l) { Sp s; s.a = a; s.l = l; return s; }
inline Sp sp_zero() { return sp(float3(0), float3(0)); }
inline Sp sp_add(Sp x, Sp y) { return sp(x.a + y.a, x.l + y.l); }
inline Sp sp_scale(Sp x, float s) { return sp(x.a * s, x.l * s); }
inline float sp_dot(Sp x, Sp y) { return dot(x.a, y.a) + dot(x.l, y.l); }
inline Sp cross_motion(Sp v, Sp m) { return sp(cross(v.a, m.a), cross(v.a, m.l) + cross(v.l, m.a)); }
inline Sp cross_force(Sp v, Sp f) { return sp(cross(v.a, f.a) + cross(v.l, f.l), cross(v.a, f.l)); }
inline float3 point_vel(Sp v, float3 p) { return v.l + cross(v.a, p); }

struct SI {
    float m;
    float3 mc;
    float3x3 io;
};
inline SI si_zero() { SI s; s.m = 0; s.mc = float3(0); s.io = float3x3(float3(0), float3(0), float3(0)); return s; }
inline SI si_from_com(float m, float3 c, float3x3 ic) {
    SI s;
    s.m = m;
    s.mc = c * m;
    s.io = ic + (mat_identity() * dot(c, c) - outer3(c, c)) * m;
    return s;
}
inline SI si_add(SI x, SI y) { SI s; s.m = x.m + y.m; s.mc = x.mc + y.mc; s.io = x.io + y.io; return s; }
inline Sp si_mul(SI s, Sp v) { return sp(s.io * v.a + cross(s.mc, v.l), v.l * s.m - cross(s.mc, v.a)); }

// dense Cholesky, lower factor in place (row-major n x n)
template <int N>
inline bool cholesky(thread float* a) {
    for (int j = 0; j < N; j++) {
        float d = a[j * N + j];
        for (int k = 0; k < j; k++) d -= a[j * N + k] * a[j * N + k];
        if (d <= 0.0f) return false;
        d = sqrt(d);
        a[j * N + j] = d;
        for (int i = j + 1; i < N; i++) {
            float s = a[i * N + j];
            for (int k = 0; k < j; k++) s -= a[i * N + k] * a[j * N + k];
            a[i * N + j] = s / d;
        }
    }
    return true;
}

template <int N>
inline void cholesky_solve(thread const float* l, thread float* x) {
    for (int i = 0; i < N; i++) {
        float s = x[i];
        for (int k = 0; k < i; k++) s -= l[i * N + k] * x[k];
        x[i] = s / l[i * N + i];
    }
    for (int i = N - 1; i >= 0; i--) {
        float s = x[i];
        for (int k = i + 1; k < N; k++) s -= l[k * N + i] * x[k];
        x[i] = s / l[i * N + i];
    }
}

// ---------------------------------------------------------------- collision

struct Contact {
    float3 pos;
    float3 n;   // from geom 1 to geom 2
    float3 t;   // preferred first tangent (0 = none)
    float depth;
    int g1;
    int g2;
};

struct Pose {
    int type;
    float3 pos;
    float3x3 mat;
    float3 size;
};

struct Hits {
    thread Contact* c;
    int n;
    int g1;
    int g2;
    bool flip;
};

inline void emit(thread Hits& h, float3 pos, float3 n, float depth, float3 t) {
    if (h.n >= MAXCON) return;
    Contact c;
    c.pos = pos;
    c.n = h.flip ? -n : n;
    c.t = t;
    c.depth = depth;
    c.g1 = h.g1;
    c.g2 = h.g2;
    h.c[h.n++] = c;
}

inline float3 closest_on_segment(float3 a, float3 b, float3 p) {
    float3 d = b - a;
    float l2 = dot(d, d);
    if (l2 < 1e-18f) return a;
    return a + d * clamp(dot(p - a, d) / l2, 0.0f, 1.0f);
}

inline void closest_segments(float3 p0, float3 p1, float3 q0, float3 q1, thread float3& x, thread float3& y) {
    float3 d1 = p1 - p0, d2 = q1 - q0, r = p0 - q0;
    float a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r);
    float s, t;
    if (a < 1e-18f && e < 1e-18f) { x = p0; y = q0; return; }
    if (a < 1e-18f) {
        s = 0; t = clamp(f / e, 0.0f, 1.0f);
    } else {
        float c = dot(d1, r);
        if (e < 1e-18f) {
            t = 0; s = clamp(-c / a, 0.0f, 1.0f);
        } else {
            float b = dot(d1, d2);
            float denom = a * e - b * b;
            s = denom > 1e-18f ? clamp((b * f - c * e) / denom, 0.0f, 1.0f) : 0.0f;
            t = (b * s + f) / e;
            if (t < 0) { t = 0; s = clamp(-c / a, 0.0f, 1.0f); }
            else if (t > 1) { t = 1; s = clamp((b - c) / a, 0.0f, 1.0f); }
        }
    }
    x = p0 + d1 * s;
    y = q0 + d2 * t;
}

inline void segment(Pose g, thread float3& a, thread float3& b) {
    float3 ax = g.mat[2] * g.size.y;
    a = g.pos - ax;
    b = g.pos + ax;
}

inline void plane_sphere(Pose pl, float3 c, float r, float3 t, thread Hits& h) {
    float3 n = pl.mat[2];
    float dist = dot(c - pl.pos, n);
    float depth = r - dist;
    if (depth > -MARGIN) emit(h, c - n * ((dist + r) * 0.5f), n, depth, t);
}

inline void plane_box(Pose pl, Pose b, thread Hits& h) {
    float3 n = pl.mat[2];
    float dist[8];
    float3 pts[8];
    int k = 0;
    for (int i = 0; i < 8; i++) {
        float3 s = float3((i & 1) ? 1 : -1, (i & 2) ? 1 : -1, (i & 4) ? 1 : -1);
        float3 corner = b.pos + b.mat * (b.size * s);
        float d = dot(corner - pl.pos, n);
        if (d < MARGIN) { dist[k] = d; pts[k] = corner; k++; }
    }
    // four deepest (selection sort)
    for (int m = 0; m < min(k, 4); m++) {
        int best = m;
        for (int i = m + 1; i < k; i++) if (dist[i] < dist[best]) best = i;
        float td = dist[m]; dist[m] = dist[best]; dist[best] = td;
        float3 tp = pts[m]; pts[m] = pts[best]; pts[best] = tp;
        emit(h, pts[m] - n * (dist[m] * 0.5f), n, -dist[m], float3(0));
    }
}

inline void plane_cylinder(Pose pl, Pose c, thread Hits& h) {
    float3 n = pl.mat[2];
    float3 axis = c.mat[2];
    float r = c.size.x, hl = c.size.y;
    float3 radial = n - axis * dot(axis, n);
    float rl = length(radial);
    float3 pts[8];
    int k = 0;
    for (int cap = -1; cap <= 1; cap += 2) {
        float3 cc = c.pos + axis * (hl * cap);
        if (rl > 0.05f) {
            pts[k++] = cc + radial * (-r / rl);
        } else {
            float3 t1, t2;
            tangents(axis, t1, t2);
            pts[k++] = cc + t1 * r;
            pts[k++] = cc - t1 * r;
            pts[k++] = cc + t2 * r;
            pts[k++] = cc - t2 * r;
        }
    }
    float dist[8];
    int m = 0;
    for (int i = 0; i < k; i++) {
        float d = dot(pts[i] - pl.pos, n);
        if (d < MARGIN) { dist[m] = d; pts[m] = pts[i]; m++; }
    }
    for (int j = 0; j < min(m, 4); j++) {
        int best = j;
        for (int i = j + 1; i < m; i++) if (dist[i] < dist[best]) best = i;
        float td = dist[j]; dist[j] = dist[best]; dist[best] = td;
        float3 tp = pts[j]; pts[j] = pts[best]; pts[best] = tp;
        emit(h, pts[j] - n * (dist[j] * 0.5f), n, -dist[j], float3(0));
    }
}

inline void sphere_sphere(float3 c1, float r1, float3 c2, float r2, thread Hits& h) {
    float3 d = c2 - c1;
    float dist = length(d);
    float depth = r1 + r2 - dist;
    if (depth <= -MARGIN) return;
    float3 n = dist > 1e-12f ? d / dist : float3(0, 0, 1);
    emit(h, c1 + n * (r1 - depth * 0.5f), n, depth, float3(0));
}

inline void sphere_box(float3 c, float r, Pose b, thread Hits& h) {
    float3 local = transpose(b.mat) * (c - b.pos);
    float3 hs = b.size;
    float3 cl = clamp(local, -hs, hs);
    float3 diff = local - cl;
    float dist = length(diff);
    if (dist > 1e-12f) {
        float depth = r - dist;
        if (depth <= -MARGIN) return;
        float3 n = b.mat * (diff * (-1.0f / dist));
        float3 surf = b.pos + b.mat * cl;
        emit(h, surf - n * (depth * 0.5f), n, depth, float3(0));
    } else {
        float best = INFINITY;
        int bi = 0;
        float bs = 1;
        for (int i = 0; i < 3; i++) {
            for (int s = -1; s <= 1; s += 2) {
                float d = hs[i] - s * local[i];
                if (d < best) { best = d; bi = i; bs = s; }
            }
        }
        float3 nl = float3(0);
        nl[bi] = -bs;
        float3 n = b.mat * nl;
        float depth = r + best;
        emit(h, c + n * (r - depth * 0.5f), n, depth, float3(0));
    }
}

inline void capsule_capsule(Pose a, Pose b, thread Hits& h) {
    float3 p0, p1, q0, q1, x, y;
    segment(a, p0, p1);
    segment(b, q0, q1);
    float ra = a.size.x, rb = b.size.x;
    closest_segments(p0, p1, q0, q1, x, y);
    sphere_sphere(x, ra, y, rb, h);
    float3 da = normalize(p1 - p0), db = normalize(q1 - q0);
    if (length(cross(da, db)) < 0.05f) {
        float3 ends[2] = {p0, p1};
        for (int i = 0; i < 2; i++) {
            float3 q = closest_on_segment(q0, q1, ends[i]);
            if (length(q - y) > 0.25f * (ra + rb) || length(ends[i] - x) > 0.25f * (ra + rb))
                sphere_sphere(ends[i], ra, q, rb, h);
        }
    }
}

inline float box_sdf(Pose b, float3 p) {
    float3 l = transpose(b.mat) * (p - b.pos);
    float3 q = fabs(l) - b.size;
    return length(max(q, float3(0))) + min(max(q.x, max(q.y, q.z)), 0.0f);
}

inline void capsule_box(Pose a, Pose b, thread Hits& h) {
    float3 p0, p1;
    segment(a, p0, p1);
    float r = a.size.x;
    const float g = 0.61803398875f;
    float lo = 0, hi = 1;
    float x1 = hi - g * (hi - lo), x2 = lo + g * (hi - lo);
    float f1 = box_sdf(b, p0 + (p1 - p0) * x1), f2 = box_sdf(b, p0 + (p1 - p0) * x2);
    for (int i = 0; i < 30; i++) {
        if (f1 < f2) {
            hi = x2; x2 = x1; f2 = f1; x1 = hi - g * (hi - lo); f1 = box_sdf(b, p0 + (p1 - p0) * x1);
        } else {
            lo = x1; x1 = x2; f1 = f2; x2 = lo + g * (hi - lo); f2 = box_sdf(b, p0 + (p1 - p0) * x2);
        }
    }
    float tmin = 0.5f * (lo + hi);
    sphere_box(p0 + (p1 - p0) * tmin, r, b, h);
    for (int e = 0; e < 2; e++) {
        float t = (float)e;
        if (fabs(t - tmin) > 0.1f && box_sdf(b, p0 + (p1 - p0) * t) < r + MARGIN) sphere_box(p0 + (p1 - p0) * t, r, b, h);
    }
}

// Sutherland-Hodgman against the half-space pn.p <= off
inline int clip_poly(thread const float3* in, int n, thread float3* out, float3 pn, float off) {
    int m = 0;
    for (int i = 0; i < n; i++) {
        float3 a = in[i], b = in[(i + 1) % n];
        float da = dot(pn, a) - off, db = dot(pn, b) - off;
        if (da <= 0) out[m++] = a;
        if ((da < 0) != (db < 0) && fabs(da - db) > 1e-15f) out[m++] = a + (b - a) * (da / (da - db));
    }
    return m;
}

inline void box_box(Pose a, Pose b, thread Hits& h) {
    float3 d = b.pos - a.pos;
    float3 ax[3] = {a.mat[0], a.mat[1], a.mat[2]};
    float3 bx[3] = {b.mat[0], b.mat[1], b.mat[2]};
    float best_o = INFINITY, face_o = INFINITY;
    float3 best_l = float3(0), face_l = float3(0);
    int best_k = -1, face_k = -1;
    for (int k = 0; k < 15; k++) {
        float3 l;
        if (k < 3) l = ax[k];
        else if (k < 6) l = bx[k - 3];
        else {
            float3 c = cross(ax[(k - 6) / 3], bx[(k - 6) % 3]);
            float n = length(c);
            if (n < 1e-6f) continue;
            l = c / n;
        }
        float pa = a.size.x * fabs(dot(ax[0], l)) + a.size.y * fabs(dot(ax[1], l)) + a.size.z * fabs(dot(ax[2], l));
        float pb = b.size.x * fabs(dot(bx[0], l)) + b.size.y * fabs(dot(bx[1], l)) + b.size.z * fabs(dot(bx[2], l));
        float o = pa + pb - fabs(dot(d, l));
        if (o < -MARGIN) return;
        if (dot(d, l) < 0) l = -l;
        if (o < best_o) { best_o = o; best_l = l; best_k = k; }
        if (k < 6 && o < face_o) { face_o = o; face_l = l; face_k = k; }
    }
    if (best_k < 0) return;
    if (best_k >= 6 && face_o < best_o * 1.05f + 1e-4f) { best_o = face_o; best_l = face_l; best_k = face_k; }
    float overlap = best_o;
    float3 n = best_l;

    if (best_k >= 6) {
        int i = (best_k - 6) / 3, j = (best_k - 6) % 3;
        float3 pa = a.pos, pb = b.pos;
        for (int m = 0; m < 3; m++) {
            if (m != i) pa += ax[m] * (a.size[m] * sign(dot(ax[m], n)));
            if (m != j) pb -= bx[m] * (b.size[m] * sign(dot(bx[m], n)));
        }
        float3 x, y;
        closest_segments(pa - ax[i] * a.size[i], pa + ax[i] * a.size[i], pb - bx[j] * b.size[j], pb + bx[j] * b.size[j], x, y);
        emit(h, (x + y) * 0.5f, n, overlap, float3(0));
        return;
    }

    bool ref_a = best_k < 3;
    Pose rf = ref_a ? a : b;
    Pose inc = ref_a ? b : a;
    float3 nref = ref_a ? n : -n;
    int fi = ref_a ? best_k : best_k - 3;
    float3 raxes[3] = {rf.mat[0], rf.mat[1], rf.mat[2]};
    float3 iaxes[3] = {inc.mat[0], inc.mat[1], inc.mat[2]};
    float3 ref_center = rf.pos + nref * rf.size[fi];
    int u = (fi + 1) % 3, v = (fi + 2) % 3;
    int j = 0;
    float bd = -1;
    for (int m = 0; m < 3; m++) {
        float dt = fabs(dot(iaxes[m], nref));
        if (dt > bd) { bd = dt; j = m; }
    }
    float s = -sign(dot(iaxes[j], nref));
    float3 ic = inc.pos + iaxes[j] * (s * inc.size[j]);
    int ku = (j + 1) % 3, kv = (j + 2) % 3;
    float3 eu = iaxes[ku] * inc.size[ku], ev = iaxes[kv] * inc.size[kv];
    float3 p1[8], p2[8];
    p1[0] = ic + eu + ev; p1[1] = ic - eu + ev; p1[2] = ic - eu - ev; p1[3] = ic + eu - ev;
    int np = 4;
    float3 sides[2] = {raxes[u], raxes[v]};
    float exts[2] = {rf.size[u], rf.size[v]};
    for (int e = 0; e < 2; e++) {
        for (int sg = -1; sg <= 1; sg += 2) {
            float3 pn = sides[e] * (float)(-sg);
            np = clip_poly(p1, np, p2, pn, dot(pn, ref_center) + exts[e]);
            for (int i = 0; i < np; i++) p1[i] = p2[i];
            if (np == 0) return;
        }
    }
    float sep[8];
    int k = 0;
    for (int i = 0; i < np; i++) {
        float sp_ = dot(p1[i] - ref_center, nref);
        if (sp_ < MARGIN) { sep[k] = sp_; p1[k] = p1[i]; k++; }
    }
    if (k > 4) {
        // deepest, farthest from it, max triangle area, then max min-distance
        int keep[4];
        int di = 0;
        for (int i = 1; i < k; i++) if (sep[i] < sep[di]) di = i;
        keep[0] = di;
        int fi2 = 0; float fd = -1;
        for (int i = 0; i < k; i++) { float dd = length_squared(p1[i] - p1[di]); if (dd > fd) { fd = dd; fi2 = i; } }
        keep[1] = fi2;
        int ti = 0; float ta = -1;
        for (int i = 0; i < k; i++) { float ar = length(cross(p1[fi2] - p1[di], p1[i] - p1[di])); if (ar > ta) { ta = ar; ti = i; } }
        keep[2] = ti;
        int qi = 0; float qd = -1;
        for (int i = 0; i < k; i++) {
            float md = min(length(p1[i] - p1[keep[0]]), min(length(p1[i] - p1[keep[1]]), length(p1[i] - p1[keep[2]])));
            if (md > qd) { qd = md; qi = i; }
        }
        keep[3] = qi;
        for (int i = 0; i < 4; i++) emit(h, p1[keep[i]] - nref * (sep[keep[i]] * 0.5f), n, -sep[keep[i]], float3(0));
    } else {
        for (int i = 0; i < k; i++) emit(h, p1[i] - nref * (sep[i] * 0.5f), n, -sep[i], float3(0));
    }
}

inline void collide_pair(Pose a, Pose b, thread Hits& h) {
    if (a.type > b.type) {
        Pose t = a; a = b; b = t;
        h.flip = !h.flip;
    }
    int ta = a.type, tb = b.type;
    if (ta == GEOM_PLANE && tb == GEOM_SPHERE) plane_sphere(a, b.pos, b.size.x, float3(0), h);
    else if (ta == GEOM_PLANE && tb == GEOM_CAPSULE) {
        float3 p0, p1;
        segment(b, p0, p1);
        plane_sphere(a, p0, b.size.x, b.mat[2], h);
        plane_sphere(a, p1, b.size.x, b.mat[2], h);
    }
    else if (ta == GEOM_PLANE && tb == GEOM_BOX) plane_box(a, b, h);
    else if (ta == GEOM_PLANE && tb == GEOM_CYLINDER) plane_cylinder(a, b, h);
    else if (ta == GEOM_SPHERE && tb == GEOM_SPHERE) sphere_sphere(a.pos, a.size.x, b.pos, b.size.x, h);
    else if (ta == GEOM_SPHERE && tb == GEOM_CAPSULE) {
        float3 p0, p1;
        segment(b, p0, p1);
        sphere_sphere(a.pos, a.size.x, closest_on_segment(p0, p1, a.pos), b.size.x, h);
    }
    else if (ta == GEOM_SPHERE && tb == GEOM_BOX) sphere_box(a.pos, a.size.x, b, h);
    else if (ta == GEOM_CAPSULE && tb == GEOM_CAPSULE) capsule_capsule(a, b, h);
    else if (ta == GEOM_CAPSULE && tb == GEOM_BOX) capsule_box(a, b, h);
    else if (ta == GEOM_BOX && tb == GEOM_BOX) box_box(a, b, h);
}

// ---------------------------------------------------------------- dynamics

inline float impedance(float pos) {
    float dmin = SOLIMP0, dmax = SOLIMP1, width = SOLIMP2, mid = SOLIMP3, power = SOLIMP4;
    float x = min(fabs(pos) / max(width, 1e-12f), 1.0f);
    if (x >= 1.0f || dmin == dmax) return dmax;
    float y = x <= mid ? pow(x, power) / pow(mid, power - 1) : 1 - pow(1 - x, power) / pow(1 - mid, power - 1);
    return clamp(dmin + y * (dmax - dmin), 1e-4f, 0.9999f);
}

inline void point_jac(thread const Sp* cdof, int body, float3 p, float3 dir, float sgn, thread float* row) {
    for (int t = 0; t < MAXDEPTH; t++) {
        if (t >= body_depth[body]) break;
        int k = body_anc[body][t];
        row[k] += sgn * dot(dir, point_vel(cdof[k], p));
    }
}

// y = M x (row-major NV x NV)
inline void mul_m(thread const float* m, thread const float* x, thread float* y) {
    for (int i = 0; i < NV; i++) {
        float s = 0;
        for (int j = 0; j < NV; j++) s += m[i * NV + j] * x[j];
        y[i] = s;
    }
}

inline float row_dot(thread const float* jac, int r, thread const float* x) {
    float s = 0;
    for (int k = 0; k < NV; k++) s += jac[r * NV + k] * x[k];
    return s;
}

// Newton constraint solve (see crates/pippin/src/newton.rs). `a` enters as the
// unconstrained acceleration and leaves constrained; qfrc_c receives J'f.
inline void newton_solve(thread const float* M, thread const float* jac, thread const float* aref, thread const float* D, int nr,
                         thread float* a, thread const float* awarm, thread float* qfrc_c) {
    float a0[NV], grad[NV], p[NV], tmp[NV], mp[NV], H[NV * NV];
    float jar[MAXROWS], jp[MAXROWS];
    bool act[MAXROWS];
    for (int k = 0; k < NV; k++) a0[k] = a[k];

    // warm start if the previous solution has lower cost
    {
        float c0 = 0, c1 = 0;
        for (int k = 0; k < NV; k++) tmp[k] = awarm[k] - a0[k];
        mul_m(M, tmp, mp);
        for (int k = 0; k < NV; k++) c1 += 0.5f * tmp[k] * mp[k];
        for (int r = 0; r < nr; r++) {
            float v0 = row_dot(jac, r, a0) - aref[r];
            float v1 = row_dot(jac, r, awarm) - aref[r];
            if (v0 < 0) c0 += 0.5f * D[r] * v0 * v0;
            if (v1 < 0) c1 += 0.5f * D[r] * v1 * v1;
        }
        if (c1 < c0) for (int k = 0; k < NV; k++) a[k] = awarm[k];
    }
    mul_m(M, a0, tmp);
    float scale = 1;
    for (int k = 0; k < NV; k++) scale += tmp[k] * tmp[k];
    scale = sqrt(scale);

    for (int r = 0; r < nr; r++) act[r] = false;
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int k = 0; k < NV; k++) tmp[k] = a[k] - a0[k];
        mul_m(M, tmp, grad);
        bool changed = iter == 0;
        for (int r = 0; r < nr; r++) {
            jar[r] = row_dot(jac, r, a) - aref[r];
            bool on = jar[r] < 0;
            changed = changed || (on != act[r]);
            act[r] = on;
            if (on) {
                float f = D[r] * jar[r];
                for (int k = 0; k < NV; k++) grad[k] += f * jac[r * NV + k];
            }
        }
        float g2 = 0;
        for (int k = 0; k < NV; k++) g2 += grad[k] * grad[k];
        if (sqrt(g2) < TOLERANCE * scale || !changed) break;

        for (int i = 0; i < NV * NV; i++) H[i] = M[i];
        for (int r = 0; r < nr; r++) {
            if (!act[r]) continue;
            for (int i = 0; i < NV; i++) {
                float ji = jac[r * NV + i];
                if (ji == 0) continue;
                float di = D[r] * ji;
                for (int j = 0; j < NV; j++) H[i * NV + j] += di * jac[r * NV + j];
            }
        }
        if (!cholesky<NV>(H)) break;
        for (int k = 0; k < NV; k++) p[k] = -grad[k];
        cholesky_solve<NV>(H, p);

        mul_m(M, p, mp);
        float mpp = 0, mpg = 0;
        for (int k = 0; k < NV; k++) { mpp += p[k] * mp[k]; mpg += tmp[k] * mp[k]; }
        for (int r = 0; r < nr; r++) jp[r] = row_dot(jac, r, p);

        float lo = 0, hi = INFINITY, al = 1;
        float d10 = mpg;
        for (int r = 0; r < nr; r++) if (jar[r] < 0) d10 += D[r] * jp[r] * jar[r];
        float ltol = 1e-6f * fabs(d10);
        for (int ls = 0; ls < 30; ls++) {
            float d1 = al * mpp + mpg, d2 = mpp;
            for (int r = 0; r < nr; r++) {
                float v = jar[r] + al * jp[r];
                if (v < 0) { float dq = D[r] * jp[r]; d1 += dq * v; d2 += dq * jp[r]; }
            }
            if (fabs(d1) <= ltol) break;
            if (d1 < 0) lo = al; else hi = al;
            float next = al - d1 / max(d2, 1e-30f);
            if (next <= lo || next >= hi) next = isinf(hi) ? 2 * max(al, lo) : 0.5f * (lo + hi);
            if (fabs(next - al) <= 1e-7f * max(fabs(al), 1.0f)) { al = next; break; }
            al = next;
        }
        for (int k = 0; k < NV; k++) a[k] += al * p[k];
    }

    for (int k = 0; k < NV; k++) qfrc_c[k] = 0;
    for (int r = 0; r < nr; r++) {
        float v = row_dot(jac, r, a) - aref[r];
        if (v < 0) {
            float f = -D[r] * v;
            for (int k = 0; k < NV; k++) qfrc_c[k] += f * jac[r * NV + k];
        }
    }
}

inline void step_one(thread float* q, thread float* v, thread const float* u, thread float* awarm) {
    // ---- kinematics ----
    float3 xpos[NBODY];
    float4 xquat[NBODY];
    float3x3 xmat[NBODY];
    float3 xipos[NBODY];
    SI cinert[NBODY];
    Sp cdof[NV];
    xpos[0] = float3(0);
    xquat[0] = float4(1, 0, 0, 0);
    xmat[0] = mat_identity();
    xipos[0] = float3(0);
    cinert[0] = si_zero();
    #pragma unroll
    for (int b = 1; b < NBODY; b++) {
        int p = body_parent[b];
        float3 pos = xpos[p] + xmat[p] * body_pos[b];
        float4 quat = qmul(xquat[p], body_quat[b]);
        #pragma unroll
        for (int j = body_jntadr[b]; j < body_jntadr[b] + body_jntnum[b]; j++) {
            int qa = jnt_qposadr[j], da = jnt_dofadr[j];
            int t = jnt_type[j];
            if (t == JNT_FREE) {
                pos = float3(q[qa], q[qa + 1], q[qa + 2]);
                quat = qnormalize(float4(q[qa + 3], q[qa + 4], q[qa + 5], q[qa + 6]));
                float3x3 r = qmat(quat);
                for (int k = 0; k < 3; k++) {
                    float3 e = float3(0);
                    e[k] = 1;
                    cdof[da + k] = sp(float3(0), e);
                    cdof[da + 3 + k] = sp(r[k], cross(pos, r[k]));
                }
            } else if (t == JNT_BALL) {
                float3 anchor = pos + qmat(quat) * jnt_pos[j];
                quat = qmul(quat, qnormalize(float4(q[qa], q[qa + 1], q[qa + 2], q[qa + 3])));
                float3x3 r = qmat(quat);
                pos = anchor - r * jnt_pos[j];
                for (int k = 0; k < 3; k++) cdof[da + k] = sp(r[k], cross(anchor, r[k]));
            } else if (t == JNT_HINGE) {
                float3x3 r = qmat(quat);
                float3 anchor = pos + r * jnt_pos[j];
                float3 axis = r * jnt_axis[j];
                quat = qmul(quat, qaxisangle(jnt_axis[j], q[qa] - qpos0[qa]));
                pos = anchor - qmat(quat) * jnt_pos[j];
                cdof[da] = sp(axis, cross(anchor, axis));
            } else {
                float3 axis = qmat(quat) * jnt_axis[j];
                pos += axis * (q[qa] - qpos0[qa]);
                cdof[da] = sp(float3(0), axis);
            }
        }
        quat = qnormalize(quat);
        xpos[b] = pos;
        xquat[b] = quat;
        xmat[b] = qmat(quat);
        xipos[b] = pos + xmat[b] * body_ipos[b];
        cinert[b] = si_from_com(body_mass[b], xipos[b], xmat[b] * body_inertia[b] * transpose(xmat[b]));
    }

    { float cs = 0; for (int i = 0; i < NBODY; i++) cs += xpos[i].x + cinert[i].m + cinert[i].io[0][0]; for (int i = 0; i < NV; i++) cs += cdof[i].a.x + cdof[i].l.y; STAGE(1, cs) }
    // ---- mass matrix (CRBA) ----
    float M[NV * NV];
    {
        SI crb[NBODY];
        for (int b = 0; b < NBODY; b++) crb[b] = cinert[b];
        #pragma unroll
        for (int b = NBODY - 1; b > 0; b--) crb[body_parent[b]] = si_add(crb[body_parent[b]], crb[b]);
        for (int i = 0; i < NV * NV; i++) M[i] = 0;
        #pragma unroll
        for (int i = 0; i < NV; i++) {
            Sp f = si_mul(crb[dof_body[i]], cdof[i]);
            #pragma unroll
            for (int t = 0; t < MAXDEPTH; t++) {
                if (t >= dof_depth[i]) break;
                int j = dof_anc[i][t];
                float val = sp_dot(cdof[j], f);
                M[i * NV + j] = val;
                M[j * NV + i] = val;
            }
            M[i * NV + i] += dof_armature[i];
        }
    }

    { float cs = 0; for (int i = 0; i < NV * NV; i++) cs += M[i]; STAGE(2, cs) }
    // ---- velocities and bias forces (RNE) ----
    float f[NV];
    {
        Sp cvel[NBODY], cdofdot[NV];
        cvel[0] = sp_zero();
        #pragma unroll
        for (int b = 1; b < NBODY; b++) {
            Sp vb = cvel[body_parent[b]];
            #pragma unroll
            for (int j = body_jntadr[b]; j < body_jntadr[b] + body_jntnum[b]; j++) {
                int da = jnt_dofadr[j], t = jnt_type[j];
                if (t == JNT_FREE || t == JNT_BALL) {
                    int lin = t == JNT_FREE ? 3 : 0;
                    int rot = da + lin;
                    for (int k = 0; k < lin; k++) {
                        cdofdot[da + k] = sp_zero();
                        vb = sp_add(vb, sp_scale(cdof[da + k], v[da + k]));
                    }
                    for (int k = 0; k < 3; k++) cdofdot[rot + k] = cross_motion(vb, cdof[rot + k]);
                    for (int k = 0; k < 3; k++) vb = sp_add(vb, sp_scale(cdof[rot + k], v[rot + k]));
                } else {
                    cdofdot[da] = cross_motion(vb, cdof[da]);
                    vb = sp_add(vb, sp_scale(cdof[da], v[da]));
                }
            }
            cvel[b] = vb;
        }
        Sp cacc[NBODY], cfrc[NBODY];
        cacc[0] = sp(float3(0), -GRAVITY);
        cfrc[0] = sp_zero();
        #pragma unroll
        for (int b = 1; b < NBODY; b++) {
            Sp ab = cacc[body_parent[b]];
            for (int k = body_dofadr[b]; k < body_dofadr[b] + body_dofnum[b]; k++) ab = sp_add(ab, sp_scale(cdofdot[k], v[k]));
            cacc[b] = ab;
            Sp hb = si_mul(cinert[b], cvel[b]);
            cfrc[b] = sp_add(si_mul(cinert[b], ab), cross_force(cvel[b], hb));
        }
        #pragma unroll
        for (int b = NBODY - 1; b > 0; b--) cfrc[body_parent[b]] = sp_add(cfrc[body_parent[b]], cfrc[b]);

        // smooth force = passive + actuator - bias
        for (int k = 0; k < NV; k++) f[k] = -dof_damping[k] * v[k] - sp_dot(cdof[k], cfrc[dof_body[k]]);
        for (int j = 0; j < NJNT; j++) {
            if (jnt_stiffness[j] != 0 && (jnt_type[j] == JNT_HINGE || jnt_type[j] == JNT_SLIDE))
                f[jnt_dofadr[j]] -= jnt_stiffness[j] * (q[jnt_qposadr[j]] - jnt_springref[j]);
        }
        for (int i = 0; i < NU; i++) {
            int j = act_joint[i];
            float gear = act_gear[i];
            float ctrl = u[i];
            if (act_ctrllimited[i]) ctrl = clamp(ctrl, act_ctrlrange[i].x, act_ctrlrange[i].y);
            float3 bias = act_bias[i];
            float force = act_gain[i] * ctrl + bias.x + bias.y * gear * q[jnt_qposadr[j]] + bias.z * gear * v[jnt_dofadr[j]];
            if (act_forcelimited[i]) force = clamp(force, act_forcerange[i].x, act_forcerange[i].y);
            f[jnt_dofadr[j]] += gear * force;
        }
    }

    { float cs = 0; for (int i = 0; i < NV; i++) cs += f[i]; for (int i = 0; i < NV * NV; i++) cs += M[i]; STAGE(3, cs) }
    // ---- unconstrained acceleration ----
    float L[NV * NV];
    float a[NV];
    for (int i = 0; i < NV * NV; i++) L[i] = M[i];
    cholesky<NV>(L);
    for (int k = 0; k < NV; k++) a[k] = f[k];
    cholesky_solve<NV>(L, a);

    { float cs = 0; for (int i = 0; i < NV; i++) cs += a[i]; for (int i = 0; i < NV * NV; i++) cs += M[i]; STAGE(4, cs) }
    // ---- collision ----
    Contact con[MAXCON];
    int ncon = 0;
    {
        float3 gpos[NGEOM];
        float3x3 gmat[NGEOM];
        for (int g = 0; g < NGEOM; g++) {
            int b = geom_body[g];
            gpos[g] = xpos[b] + xmat[b] * geom_pos[g];
            gmat[g] = xmat[b] * qmat(geom_quat[g]);
        }
        for (int pi = 0; pi < NPAIR; pi++) {
            int g1 = pair_geom[pi].x, g2 = pair_geom[pi].y;
            float r = geom_rbound[g1] + geom_rbound[g2] + MARGIN;
            if (!isinf(r) && length_squared(gpos[g1] - gpos[g2]) > r * r) continue;
            Pose a1, a2;
            a1.type = geom_type[g1]; a1.pos = gpos[g1]; a1.mat = gmat[g1]; a1.size = geom_size[g1];
            a2.type = geom_type[g2]; a2.pos = gpos[g2]; a2.mat = gmat[g2]; a2.size = geom_size[g2];
            Hits h;
            h.c = con; h.n = ncon; h.g1 = g1; h.g2 = g2; h.flip = false;
            collide_pair(a1, a2, h);
            ncon = h.n;
        }
    }

    { float cs = 0; for (int i = 0; i < NV; i++) cs += a[i]; for (int i = 0; i < NV * NV; i++) cs += M[i]; for (int i = 0; i < ncon; i++) cs += con[i].depth + con[i].pos.x + con[i].n.y; STAGE(5, cs) }
    // ---- constraint rows ----
    float jac[MAXROWS * NV];
    float aref[MAXROWS], D[MAXROWS];
    int nr = 0;
    {
        const float tc = max(SOLREF0, 2.0f * TIMESTEP);
        const float kb = 2.0f / (SOLIMP1 * tc);
        const float kk = 1.0f / (SOLIMP1 * SOLIMP1 * tc * tc * SOLREF1 * SOLREF1);
        for (int ci = 0; ci < ncon; ci++) {
            Contact c = con[ci];
            if (c.depth <= 0 || nr + 4 > MAXROWS) continue;
            int b1 = geom_body[c.g1], b2 = geom_body[c.g2];
            float mu = max(geom_friction[c.g1], geom_friction[c.g2]);
            float3 t1, t2;
            float3 tt = c.t - c.n * dot(c.n, c.t);
            if (length_squared(tt) > 1e-12f) { t1 = normalize(tt); t2 = cross(c.n, t1); }
            else tangents(c.n, t1, t2);
            float jn[NV], jt1[NV], jt2[NV];
            for (int k = 0; k < NV; k++) { jn[k] = 0; jt1[k] = 0; jt2[k] = 0; }
            point_jac(cdof, b2, c.pos, c.n, 1, jn);  point_jac(cdof, b1, c.pos, c.n, -1, jn);
            point_jac(cdof, b2, c.pos, t1, 1, jt1);  point_jac(cdof, b1, c.pos, t1, -1, jt1);
            point_jac(cdof, b2, c.pos, t2, 1, jt2);  point_jac(cdof, b1, c.pos, t2, -1, jt2);
            float tran = body_invweight[b1] + body_invweight[b2];
            float diag = 2 * mu * mu * (1 + mu * mu) * tran / IMPRATIO;
            float imp = impedance(-c.depth);
            float Dr = 1.0f / max((1 - imp) / imp * diag, 1e-15f);
            for (int e = 0; e < 4; e++) {
                float sgn = (e & 1) ? -1.0f : 1.0f;
                thread float* jt = e < 2 ? jt1 : jt2;
                float vel = 0;
                for (int k = 0; k < NV; k++) {
                    float jv = jn[k] + sgn * mu * jt[k];
                    jac[nr * NV + k] = jv;
                    vel += jv * v[k];
                }
                aref[nr] = -kb * vel - kk * imp * (-c.depth);
                D[nr] = Dr;
                nr++;
            }
        }
        for (int j = 0; j < NJNT; j++) {
            if (!jnt_limited[j]) continue;
            int da = jnt_dofadr[j];
            float qj = q[jnt_qposadr[j]];
            for (int side = 0; side < 2; side++) {
                float sep = side == 0 ? qj - jnt_range[j].x : jnt_range[j].y - qj;
                float sgn = side == 0 ? 1.0f : -1.0f;
                if (sep >= 0 || nr >= MAXROWS) continue;
                for (int k = 0; k < NV; k++) jac[nr * NV + k] = 0;
                jac[nr * NV + da] = sgn;
                float imp = impedance(sep);
                aref[nr] = -kb * sgn * v[da] - kk * imp * sep;
                D[nr] = 1.0f / max((1 - imp) / imp * dof_invweight[da], 1e-15f);
                nr++;
            }
        }
    }

    { float cs = 0; for (int i = 0; i < NV; i++) cs += a[i]; for (int i = 0; i < NV * NV; i++) cs += M[i]; for (int r = 0; r < nr; r++) { cs += aref[r] + D[r]; for (int k = 0; k < NV; k++) cs += jac[r * NV + k]; } STAGE(6, cs) }
    // ---- solve, implicit damping, integrate ----
    float qfrc_c[NV];
    for (int k = 0; k < NV; k++) qfrc_c[k] = 0;
    if (nr > 0) newton_solve(M, jac, aref, D, nr, a, awarm, qfrc_c);
    { float cs = 0; for (int i = 0; i < NV; i++) cs += a[i] + qfrc_c[i] + f[i]; for (int i = 0; i < NV * NV; i++) cs += M[i]; STAGE(7, cs) }
    if (DAMPED) {
        for (int i = 0; i < NV * NV; i++) L[i] = M[i];
        for (int k = 0; k < NV; k++) L[k * NV + k] += TIMESTEP * dof_damping[k];
        cholesky<NV>(L);
        for (int k = 0; k < NV; k++) a[k] = f[k] + qfrc_c[k];
        cholesky_solve<NV>(L, a);
    }
    for (int k = 0; k < NV; k++) {
        awarm[k] = a[k];
        v[k] += TIMESTEP * a[k];
    }
    for (int j = 0; j < NJNT; j++) {
        int qa = jnt_qposadr[j], da = jnt_dofadr[j];
        int t = jnt_type[j];
        if (t == JNT_FREE) {
            for (int k = 0; k < 3; k++) q[qa + k] += TIMESTEP * v[da + k];
            float4 qq = qintegrate_local(float4(q[qa + 3], q[qa + 4], q[qa + 5], q[qa + 6]), float3(v[da + 3], v[da + 4], v[da + 5]), TIMESTEP);
            q[qa + 3] = qq[0]; q[qa + 4] = qq[1]; q[qa + 5] = qq[2]; q[qa + 6] = qq[3];
        } else if (t == JNT_BALL) {
            float4 qq = qintegrate_local(float4(q[qa], q[qa + 1], q[qa + 2], q[qa + 3]), float3(v[da], v[da + 1], v[da + 2]), TIMESTEP);
            q[qa] = qq[0]; q[qa + 1] = qq[1]; q[qa + 2] = qq[2]; q[qa + 3] = qq[3];
        } else {
            q[qa] += TIMESTEP * v[da];
        }
    }
}

kernel void step_kernel(device float* qpos [[buffer(0)]],
                        device float* qvel [[buffer(1)]],
                        device const float* ctrl [[buffer(2)]],
                        device float* qacc [[buffer(3)]],
                        constant uint& nenv [[buffer(4)]],
                        constant uint& nstep [[buffer(5)]],
                        uint tid [[thread_position_in_grid]]) {
    if (tid >= nenv) return;
    float q[NQ], v[NV], u[NU_ALLOC], aw[NV];
    for (int k = 0; k < NQ; k++) q[k] = qpos[tid * NQ + k];
    for (int k = 0; k < NV; k++) { v[k] = qvel[tid * NV + k]; aw[k] = qacc[tid * NV + k]; }
    for (int k = 0; k < NU; k++) u[k] = ctrl[tid * NU + k];
    for (uint s = 0; s < nstep; s++) step_one(q, v, u, aw);
    for (int k = 0; k < NQ; k++) qpos[tid * NQ + k] = q[k];
    for (int k = 0; k < NV; k++) { qvel[tid * NV + k] = v[k]; qacc[tid * NV + k] = aw[k]; }
}
