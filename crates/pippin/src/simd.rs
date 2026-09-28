//! Vectorized kernels for the solver's hot loops.
//!
//! Rust (like C without -ffast-math) may not reassociate floating-point sums,
//! so the compiler leaves dot products scalar: one fused multiply-add per
//! cycle, serialized on its latency. These kernels use NEON (2 x f64 lanes)
//! with four independent accumulators, as Apple's CPU optimization guide
//! recommends, to keep the FMA pipes busy. Results differ from a sequential
//! sum only by rounding.

use crate::math::Real;

/// sum_k a[k] * b[k]
#[inline]
pub fn dot(a: &[Real], b: &[Real]) -> Real {
    let n = a.len().min(b.len());
    if n < 8 {
        // short vectors (small robots): accumulator setup costs more than it saves
        let mut s = 0.0;
        for k in 0..n {
            s += a[k] * b[k];
        }
        return s;
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc = [vdupq_n_f64(0.0); 4];
        let mut k = 0;
        while k + 8 <= n {
            for (j, s) in acc.iter_mut().enumerate() {
                *s = vfmaq_f64(*s, vld1q_f64(pa.add(k + 2 * j)), vld1q_f64(pb.add(k + 2 * j)));
            }
            k += 8;
        }
        while k + 2 <= n {
            acc[0] = vfmaq_f64(acc[0], vld1q_f64(pa.add(k)), vld1q_f64(pb.add(k)));
            k += 2;
        }
        let v = vaddq_f64(vaddq_f64(acc[0], acc[1]), vaddq_f64(acc[2], acc[3]));
        let mut s = vaddvq_f64(v);
        if k < n {
            s += a[k] * b[k];
        }
        s
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut s = [0.0; 4];
        let mut k = 0;
        while k + 4 <= n {
            for j in 0..4 {
                s[j] += a[k + j] * b[k + j];
            }
            k += 4;
        }
        let mut t = (s[0] + s[1]) + (s[2] + s[3]);
        while k < n {
            t += a[k] * b[k];
            k += 1;
        }
        t
    }
}

/// y[k] += alpha * x[k]  (no reduction: the compiler vectorizes this itself)
#[inline]
pub fn axpy(alpha: Real, x: &[Real], y: &mut [Real]) {
    for (yk, xk) in y.iter_mut().zip(x) {
        *yk += alpha * xk;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_scalar() {
        for n in 0..40 {
            let a: Vec<Real> = (0..n).map(|i| (i as Real * 0.37).sin()).collect();
            let b: Vec<Real> = (0..n).map(|i| (i as Real * 0.91).cos()).collect();
            let s: Real = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            assert!((dot(&a, &b) - s).abs() < 1e-12, "n={n}");
        }
    }
}
