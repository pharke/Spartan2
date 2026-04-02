// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Radix-2 NTT utilities over prime fields.
use ff::PrimeField;
use std::collections::HashMap;
use thiserror::Error;

/// Default threshold for switching polynomial multiplication to naive mode.
///
/// When `a.len().min(b.len()) <= POLY_MUL_NAIVE_THRESHOLD`, the adaptive
/// routine uses schoolbook multiplication instead of NTT.
pub const POLY_MUL_NAIVE_THRESHOLD: usize = 64;
/// Default threshold for switching to direct root expansion.
pub const ROOTS_NAIVE_THRESHOLD: usize = 64;

/// Reusable workspace for repeated polynomial multiplications.
///
/// This reduces allocation churn by reusing temporary buffers and cached
/// stage twiddles across calls.
#[derive(Clone, Debug)]
pub struct PolyMulWorkspace<F: PrimeField> {
  a_pad: Vec<F>,
  b_pad: Vec<F>,
  stage_twiddles: HashMap<u32, (Vec<F>, Vec<F>)>,
}

impl<F: PrimeField> Default for PolyMulWorkspace<F> {
  fn default() -> Self {
    Self {
      a_pad: Vec::new(),
      b_pad: Vec::new(),
      stage_twiddles: HashMap::new(),
    }
  }
}

/// Errors returned by radix-2 NTT routines.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum NttError {
  /// Input length must be a power of two.
  #[error("NTT input length must be a power of two")]
  NonPowerOfTwoLength,
  /// Input length exceeds the field's 2-adicity.
  #[error("NTT input length exceeds field two-adicity")]
  UnsupportedLength,
  /// Input length is too large to represent safely in `u64`.
  #[error("NTT input length is too large")]
  LengthTooLarge,
}

/// Computes an in-place radix-2 NTT over `values`.
///
/// The transform length is `values.len() = 2^k`, and requires `k <= F::S`.
pub fn radix2_ntt<F: PrimeField>(values: &mut [F]) -> Result<(), NttError> {
  ntt_in_place(values, false)
}

/// Computes an in-place inverse radix-2 NTT over `values`.
///
/// This is the inverse of [`radix2_ntt`], including scaling by `n^{-1}`.
pub fn radix2_intt<F: PrimeField>(values: &mut [F]) -> Result<(), NttError> {
  ntt_in_place(values, true)
}

/// Multiplies two dense polynomials using radix-2 NTT.
///
/// Inputs are coefficient vectors in little-endian order:
/// `a(x) = a[0] + a[1]x + ...`, `b(x) = b[0] + b[1]x + ...`.
///
/// Returns the coefficient vector of `c(x) = a(x) * b(x)` in the same order.
pub fn poly_mul_ntt<F: PrimeField>(a: &[F], b: &[F]) -> Result<Vec<F>, NttError> {
  let mut workspace = PolyMulWorkspace::<F>::default();
  poly_mul_ntt_with_workspace(a, b, &mut workspace)
}

/// Multiplies two dense polynomials using radix-2 NTT and a reusable workspace.
pub fn poly_mul_ntt_with_workspace<F: PrimeField>(
  a: &[F],
  b: &[F],
  workspace: &mut PolyMulWorkspace<F>,
) -> Result<Vec<F>, NttError> {
  if a.is_empty() || b.is_empty() {
    return Ok(Vec::new());
  }

  let out_len = a.len() + b.len() - 1;
  let n = out_len.next_power_of_two();

  if n.trailing_zeros() > F::S {
    return Err(NttError::UnsupportedLength);
  }

  prepare_buffer(&mut workspace.a_pad, n);
  prepare_buffer(&mut workspace.b_pad, n);
  workspace.a_pad[..a.len()].copy_from_slice(a);
  workspace.b_pad[..b.len()].copy_from_slice(b);

  let log_n = n.trailing_zeros();
  let (stage_roots_fwd, stage_roots_inv) = workspace
    .stage_twiddles
    .entry(log_n)
    .or_insert_with(|| precompute_stage_twiddles::<F>(n));

  ntt_in_place_with_twiddles(&mut workspace.a_pad, stage_roots_fwd, false)?;
  ntt_in_place_with_twiddles(&mut workspace.b_pad, stage_roots_fwd, false)?;

  for (x, y) in workspace.a_pad.iter_mut().zip(workspace.b_pad.iter()) {
    *x *= *y;
  }

  ntt_in_place_with_twiddles(&mut workspace.a_pad, stage_roots_inv, true)?;
  Ok(workspace.a_pad[..out_len].to_vec())
}

/// Multiplies two dense polynomials with an adaptive strategy.
///
/// This uses schoolbook multiplication for small inputs and falls back to
/// [`poly_mul_ntt`] for larger inputs.
pub fn poly_mul_adaptive<F: PrimeField>(a: &[F], b: &[F]) -> Result<Vec<F>, NttError> {
  let mut workspace = PolyMulWorkspace::<F>::default();
  poly_mul_adaptive_with_workspace(a, b, &mut workspace)
}

/// Adaptive multiplication using reusable workspace.
pub fn poly_mul_adaptive_with_workspace<F: PrimeField>(
  a: &[F],
  b: &[F],
  workspace: &mut PolyMulWorkspace<F>,
) -> Result<Vec<F>, NttError> {
  if a.len().min(b.len()) <= POLY_MUL_NAIVE_THRESHOLD {
    Ok(poly_mul_naive(a, b))
  } else {
    poly_mul_ntt_with_workspace(a, b, workspace)
  }
}

/// Builds the monic polynomial whose roots are `roots`:
/// `v(x) = \prod_i (x - roots[i])`.
///
/// Returns coefficients in little-endian order.
pub fn poly_from_roots_ntt<F: PrimeField>(roots: &[F]) -> Result<Vec<F>, NttError> {
  if roots.is_empty() {
    return Ok(vec![F::ONE]);
  }

  let mut level = roots
    .iter()
    .map(|r| vec![-*r, F::ONE])
    .collect::<Vec<Vec<F>>>();

  while level.len() > 1 {
    let mut next = Vec::with_capacity(level.len().div_ceil(2));
    for pair in level.chunks(2) {
      if pair.len() == 2 {
        next.push(poly_mul_ntt(&pair[0], &pair[1])?);
      } else {
        next.push(pair[0].clone());
      }
    }
    level = next;
  }

  Ok(level.pop().expect("non-empty by construction"))
}

/// Builds `v(x) = \prod_i (x - roots[i])` with NTT multiplications and workspace reuse.
pub fn poly_from_roots_ntt_with_workspace<F: PrimeField>(
  roots: &[F],
  workspace: &mut PolyMulWorkspace<F>,
) -> Result<Vec<F>, NttError> {
  if roots.is_empty() {
    return Ok(vec![F::ONE]);
  }

  let mut level = roots
    .iter()
    .map(|r| vec![-*r, F::ONE])
    .collect::<Vec<Vec<F>>>();

  while level.len() > 1 {
    let mut next = Vec::with_capacity(level.len().div_ceil(2));
    for pair in level.chunks(2) {
      if pair.len() == 2 {
        next.push(poly_mul_ntt_with_workspace(&pair[0], &pair[1], workspace)?);
      } else {
        next.push(pair[0].clone());
      }
    }
    level = next;
  }

  Ok(level.pop().expect("non-empty by construction"))
}

/// Builds the monic polynomial `v(x) = \prod_i (x - roots[i])` adaptively.
///
/// This keeps the same product-tree structure but uses [`poly_mul_adaptive`]
/// for each merge step, which is typically faster for small/medium inputs.
pub fn poly_from_roots_adaptive<F: PrimeField>(roots: &[F]) -> Result<Vec<F>, NttError> {
  if roots.is_empty() {
    return Ok(vec![F::ONE]);
  }

  let mut level = roots
    .iter()
    .map(|r| vec![-*r, F::ONE])
    .collect::<Vec<Vec<F>>>();

  while level.len() > 1 {
    let mut next = Vec::with_capacity(level.len().div_ceil(2));
    for pair in level.chunks(2) {
      if pair.len() == 2 {
        next.push(poly_mul_adaptive(&pair[0], &pair[1])?);
      } else {
        next.push(pair[0].clone());
      }
    }
    level = next;
  }

  Ok(level.pop().expect("non-empty by construction"))
}

/// Builds `v(x) = \prod_i (x - roots[i])` adaptively with workspace reuse.
pub fn poly_from_roots_adaptive_with_workspace<F: PrimeField>(
  roots: &[F],
  workspace: &mut PolyMulWorkspace<F>,
) -> Result<Vec<F>, NttError> {
  if roots.is_empty() {
    return Ok(vec![F::ONE]);
  }

  let mut level = roots
    .iter()
    .map(|r| vec![-*r, F::ONE])
    .collect::<Vec<Vec<F>>>();

  while level.len() > 1 {
    let mut next = Vec::with_capacity(level.len().div_ceil(2));
    for pair in level.chunks(2) {
      if pair.len() == 2 {
        next.push(poly_mul_adaptive_with_workspace(
          &pair[0], &pair[1], workspace,
        )?);
      } else {
        next.push(pair[0].clone());
      }
    }
    level = next;
  }

  Ok(level.pop().expect("non-empty by construction"))
}

/// Final optimized API for `v(x)=\prod_i (x-roots[i])`.
///
/// It combines:
/// - small-size direct expansion,
/// - product tree for larger inputs,
/// - adaptive multiplication with reusable workspace.
pub fn poly_from_roots_optimized<F: PrimeField>(roots: &[F]) -> Result<Vec<F>, NttError> {
  if roots.len() <= ROOTS_NAIVE_THRESHOLD {
    return Ok(poly_from_roots_naive_generic(roots));
  }
  poly_from_roots_adaptive(roots)
}

fn poly_mul_naive<F: PrimeField>(a: &[F], b: &[F]) -> Vec<F> {
  if a.is_empty() || b.is_empty() {
    return Vec::new();
  }
  let mut out = vec![F::ZERO; a.len() + b.len() - 1];
  for (i, ai) in a.iter().enumerate() {
    for (j, bj) in b.iter().enumerate() {
      out[i + j] += *ai * *bj;
    }
  }
  out
}

fn poly_from_roots_naive_generic<F: PrimeField>(roots: &[F]) -> Vec<F> {
  let mut poly = vec![F::ONE];
  for r in roots {
    let mut next = vec![F::ZERO; poly.len() + 1];
    for (i, c) in poly.iter().enumerate() {
      next[i] -= *c * *r;
      next[i + 1] += *c;
    }
    poly = next;
  }
  poly
}

fn prepare_buffer<F: PrimeField>(buf: &mut Vec<F>, n: usize) {
  if buf.len() < n {
    buf.resize(n, F::ZERO);
  } else {
    buf.truncate(n);
    buf[..n].fill(F::ZERO);
  }
}

fn precompute_stage_twiddles<F: PrimeField>(n: usize) -> (Vec<F>, Vec<F>) {
  let n_u64 = u64::try_from(n).expect("n fits u64 by caller checks");
  let log_n = n.trailing_zeros();
  let omega = primitive_2k_root::<F>(log_n);
  let omega_inv = omega
    .invert()
    .into_option()
    .expect("non-zero root of unity must be invertible");

  let mut stage_fwd = Vec::with_capacity(log_n as usize);
  let mut stage_inv = Vec::with_capacity(log_n as usize);
  let mut m = 1usize;
  while m < n {
    let span = m << 1;
    let step = n_u64 / u64::try_from(span).expect("span fits u64");
    stage_fwd.push(omega.pow_vartime([step]));
    stage_inv.push(omega_inv.pow_vartime([step]));
    m = span;
  }
  (stage_fwd, stage_inv)
}

fn ntt_in_place<F: PrimeField>(values: &mut [F], inverse: bool) -> Result<(), NttError> {
  let n = values.len();
  if n == 0 || !n.is_power_of_two() {
    return Err(NttError::NonPowerOfTwoLength);
  }

  let log_n = n.trailing_zeros();
  if log_n > F::S {
    return Err(NttError::UnsupportedLength);
  }

  let (stage_fwd, stage_inv) = precompute_stage_twiddles::<F>(n);
  let stage_roots = if inverse { &stage_inv } else { &stage_fwd };
  ntt_in_place_with_twiddles(values, stage_roots, inverse)
}

fn ntt_in_place_with_twiddles<F: PrimeField>(
  values: &mut [F],
  stage_roots: &[F],
  inverse: bool,
) -> Result<(), NttError> {
  let n = values.len();
  let n_u64 = u64::try_from(n).map_err(|_| NttError::LengthTooLarge)?;

  bit_reverse_permute(values);

  let mut m = 1usize;
  let mut stage_idx = 0usize;
  while m < n {
    let span = m << 1;
    let w_m = stage_roots[stage_idx];
    stage_idx += 1;

    for chunk in values.chunks_exact_mut(span) {
      let mut w = F::ONE;
      let (left, right) = chunk.split_at_mut(m);
      for j in 0..m {
        let t = w * right[j];
        let u = left[j];
        left[j] = u + t;
        right[j] = u - t;
        w *= w_m;
      }
    }
    m = span;
  }

  if inverse {
    let n_inv = F::from(n_u64)
      .invert()
      .into_option()
      .ok_or(NttError::UnsupportedLength)?;
    for value in values.iter_mut() {
      *value *= n_inv;
    }
  }

  Ok(())
}

fn primitive_2k_root<F: PrimeField>(k: u32) -> F {
  let mut root = F::ROOT_OF_UNITY;
  for _ in k..F::S {
    root = root.square();
  }
  root
}

fn bit_reverse_permute<F>(values: &mut [F]) {
  let n = values.len();
  if n <= 1 {
    return;
  }
  let bits = n.trailing_zeros();
  for i in 0..n {
    let j = i.reverse_bits() >> (usize::BITS - bits);
    if i < j {
      values.swap(i, j);
    }
  }
}

#[cfg(test)]
mod tests {
  use super::{
    PolyMulWorkspace, poly_from_roots_adaptive, poly_from_roots_ntt, poly_from_roots_optimized,
    poly_mul_adaptive, poly_mul_ntt, poly_mul_ntt_with_workspace, radix2_intt, radix2_ntt,
  };
  use crate::provider::pasta::pallas;
  use ff::Field;
  use rand::{SeedableRng, rngs::OsRng, rngs::StdRng};
  use std::time::Instant;

  #[test]
  fn test_ntt_roundtrip_pallas_scalar() {
    for log_n in 0..=10 {
      let n = 1usize << log_n;
      let mut values = (0..n)
        .map(|_| pallas::Scalar::random(&mut OsRng))
        .collect::<Vec<_>>();
      let original = values.clone();

      radix2_ntt(&mut values).unwrap();
      radix2_intt(&mut values).unwrap();

      assert_eq!(values, original);
    }
  }

  #[test]
  fn test_ntt_delta_vector() {
    let n = 8usize;
    let mut values = vec![pallas::Scalar::ZERO; n];
    values[0] = pallas::Scalar::ONE;

    radix2_ntt(&mut values).unwrap();
    assert!(values.iter().all(|x| *x == pallas::Scalar::ONE));
  }

  fn poly_from_roots_naive(roots: &[pallas::Scalar]) -> Vec<pallas::Scalar> {
    super::poly_from_roots_naive_generic(roots)
  }

  #[test]
  fn test_poly_mul_ntt_matches_naive() {
    for (na, nb) in [(1usize, 1usize), (3, 5), (8, 8), (17, 9), (32, 31)] {
      let a = (0..na)
        .map(|_| pallas::Scalar::random(&mut OsRng))
        .collect::<Vec<_>>();
      let b = (0..nb)
        .map(|_| pallas::Scalar::random(&mut OsRng))
        .collect::<Vec<_>>();

      let got = poly_mul_ntt(&a, &b).unwrap();
      let expected = super::poly_mul_naive(&a, &b);
      assert_eq!(got, expected);
    }
  }

  #[test]
  fn test_poly_from_roots_ntt_small() {
    let roots = vec![
      pallas::Scalar::from(1),
      pallas::Scalar::from(2),
      pallas::Scalar::from(3),
    ];
    let coeffs = poly_from_roots_ntt(&roots).unwrap();
    // (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
    let expected = vec![
      -pallas::Scalar::from(6),
      pallas::Scalar::from(11),
      -pallas::Scalar::from(6),
      pallas::Scalar::ONE,
    ];
    assert_eq!(coeffs, expected);
  }

  #[test]
  fn test_poly_from_roots_ntt_vanishes_on_roots() {
    let roots = (0..20)
      .map(|_| pallas::Scalar::random(&mut OsRng))
      .collect::<Vec<_>>();
    let coeffs = poly_from_roots_ntt(&roots).unwrap();

    for r in roots {
      let mut acc = pallas::Scalar::ZERO;
      let mut pow = pallas::Scalar::ONE;
      for c in &coeffs {
        acc += *c * pow;
        pow *= r;
      }
      assert_eq!(acc, pallas::Scalar::ZERO);
    }
  }

  #[test]
  fn test_poly_mul_adaptive_matches_naive() {
    for (na, nb) in [(8usize, 8usize), (64, 64), (128, 128), (257, 113)] {
      let a = (0..na)
        .map(|_| pallas::Scalar::random(&mut OsRng))
        .collect::<Vec<_>>();
      let b = (0..nb)
        .map(|_| pallas::Scalar::random(&mut OsRng))
        .collect::<Vec<_>>();
      let got = poly_mul_adaptive(&a, &b).unwrap();
      let expected = super::poly_mul_naive(&a, &b);
      assert_eq!(got, expected);
    }
  }

  #[test]
  fn test_poly_mul_ntt_with_workspace_matches_ntt() {
    let mut workspace = PolyMulWorkspace::<pallas::Scalar>::default();
    let a = (0..257)
      .map(|_| pallas::Scalar::random(&mut OsRng))
      .collect::<Vec<_>>();
    let b = (0..193)
      .map(|_| pallas::Scalar::random(&mut OsRng))
      .collect::<Vec<_>>();
    let got = poly_mul_ntt_with_workspace(&a, &b, &mut workspace).unwrap();
    let expected = poly_mul_ntt(&a, &b).unwrap();
    assert_eq!(got, expected);
  }

  #[test]
  fn test_poly_from_roots_adaptive_matches_ntt() {
    let roots = (0..257)
      .map(|_| pallas::Scalar::random(&mut OsRng))
      .collect::<Vec<_>>();
    let got = poly_from_roots_adaptive(&roots).unwrap();
    let expected = poly_from_roots_ntt(&roots).unwrap();
    assert_eq!(got, expected);
  }

  #[test]
  fn test_poly_from_roots_optimized_matches_naive() {
    for n in [0usize, 1, 2, 3, 8, 33, 129, 257] {
      let roots = (0..n)
        .map(|_| pallas::Scalar::random(&mut OsRng))
        .collect::<Vec<_>>();
      let got = poly_from_roots_optimized(&roots).unwrap();
      let expected = poly_from_roots_naive(&roots);
      assert_eq!(got, expected);
    }
  }

  /// Large-scale perf smoke test (run manually):
  /// cargo test provider::ntt::tests::test_ntt_large_perf -- --ignored --nocapture
  #[test]
  #[ignore]
  fn test_ntt_large_perf() {
    let mut rng = StdRng::seed_from_u64(42);

    let roots_n = 1usize << 14;
    let roots = (0..roots_n)
      .map(|_| pallas::Scalar::random(&mut rng))
      .collect::<Vec<_>>();

    let t0 = Instant::now();
    let coeffs = poly_from_roots_ntt(&roots).unwrap();
    let roots_elapsed = t0.elapsed();
    assert_eq!(coeffs.len(), roots_n + 1);
    println!(
      "poly_from_roots_ntt n={} elapsed_ms={}",
      roots_n,
      roots_elapsed.as_millis()
    );

    let naive_roots_n = 1024usize;
    let roots_small = roots[..naive_roots_n].to_vec();
    let t0b = Instant::now();
    let coeffs_ntt_small = poly_from_roots_ntt(&roots_small).unwrap();
    let ntt_roots_elapsed = t0b.elapsed();
    let t0c = Instant::now();
    let coeffs_naive_small = poly_from_roots_naive(&roots_small);
    let naive_roots_elapsed = t0c.elapsed();
    let t0d = Instant::now();
    let coeffs_adaptive_small = poly_from_roots_adaptive(&roots_small).unwrap();
    let adaptive_roots_elapsed = t0d.elapsed();
    assert_eq!(coeffs_ntt_small, coeffs_naive_small);
    assert_eq!(coeffs_adaptive_small, coeffs_naive_small);
    println!(
      "poly_from_roots size={} ntt_ms={} adaptive_ms={} naive_ms={} adaptive_vs_naive_x={:.2}",
      naive_roots_n,
      ntt_roots_elapsed.as_millis(),
      adaptive_roots_elapsed.as_millis(),
      naive_roots_elapsed.as_millis(),
      naive_roots_elapsed.as_secs_f64() / adaptive_roots_elapsed.as_secs_f64()
    );

    let mul_n = 2048usize;
    let a = (0..mul_n)
      .map(|_| pallas::Scalar::random(&mut rng))
      .collect::<Vec<_>>();
    let b = (0..mul_n)
      .map(|_| pallas::Scalar::random(&mut rng))
      .collect::<Vec<_>>();

    let t1 = Instant::now();
    let ntt_prod = poly_mul_ntt(&a, &b).unwrap();
    let ntt_elapsed = t1.elapsed();

    let mut workspace = PolyMulWorkspace::<pallas::Scalar>::default();
    let t1w = Instant::now();
    let ntt_ws_prod = poly_mul_ntt_with_workspace(&a, &b, &mut workspace).unwrap();
    let ntt_ws_elapsed = t1w.elapsed();

    let t1b = Instant::now();
    let adaptive_prod = poly_mul_adaptive(&a, &b).unwrap();
    let adaptive_elapsed = t1b.elapsed();

    let t2 = Instant::now();
    let naive_prod = super::poly_mul_naive(&a, &b);
    let naive_elapsed = t2.elapsed();

    assert_eq!(ntt_prod, naive_prod);
    assert_eq!(ntt_ws_prod, naive_prod);
    assert_eq!(adaptive_prod, naive_prod);
    println!(
      "poly_mul size={} ntt_ms={} ntt_ws_ms={} adaptive_ms={} naive_ms={} adaptive_vs_naive_x={:.2}",
      mul_n,
      ntt_elapsed.as_millis(),
      ntt_ws_elapsed.as_millis(),
      adaptive_elapsed.as_millis(),
      naive_elapsed.as_millis(),
      naive_elapsed.as_secs_f64() / adaptive_elapsed.as_secs_f64()
    );
  }

  /// Compare optimized-vs-naive speedup over different sizes.
  /// Run manually:
  /// cargo test provider::ntt::tests::test_poly_from_roots_speed_table -- --ignored --nocapture
  #[test]
  #[ignore]
  fn test_poly_from_roots_speed_table() {
    let mut rng = StdRng::seed_from_u64(20260402);
    let sizes = [128usize, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768];
    println!("n,optimized_ms,naive_ms,speedup_x");

    for &n in &sizes {
      let roots = (0..n)
        .map(|_| pallas::Scalar::random(&mut rng))
        .collect::<Vec<_>>();

      let t_opt = Instant::now();
      let opt = poly_from_roots_optimized(&roots).unwrap();
      let opt_d = t_opt.elapsed();

      let t_naive = Instant::now();
      let naive = poly_from_roots_naive(&roots);
      let naive_d = t_naive.elapsed();

      assert_eq!(opt, naive);
      println!(
        "{},{},{},{:.2}",
        n,
        opt_d.as_millis(),
        naive_d.as_millis(),
        naive_d.as_secs_f64() / opt_d.as_secs_f64()
      );
    }
  }
}
