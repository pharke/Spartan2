// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Radix-2 NTT utilities over prime fields.
use ff::PrimeField;
use thiserror::Error;

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

fn ntt_in_place<F: PrimeField>(values: &mut [F], inverse: bool) -> Result<(), NttError> {
  let n = values.len();
  if n == 0 || !n.is_power_of_two() {
    return Err(NttError::NonPowerOfTwoLength);
  }

  let log_n = n.trailing_zeros();
  if log_n > F::S {
    return Err(NttError::UnsupportedLength);
  }

  bit_reverse_permute(values);

  let omega = primitive_2k_root::<F>(log_n);
  let omega = if inverse {
    omega.invert().into_option().ok_or(NttError::UnsupportedLength)?
  } else {
    omega
  };

  let n_u64 = u64::try_from(n).map_err(|_| NttError::LengthTooLarge)?;
  let mut m = 1usize;
  while m < n {
    let span = m << 1;
    let step = n_u64 / u64::try_from(span).map_err(|_| NttError::LengthTooLarge)?;
    let w_m = omega.pow_vartime([step]);

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
  use super::{radix2_intt, radix2_ntt};
  use crate::provider::pasta::pallas;
  use ff::Field;
  use rand::rngs::OsRng;

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
}
