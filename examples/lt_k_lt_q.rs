// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! `k < q` 示例（可运行的 Spartan ZK SNARK 电路）
//!
//! - `k` 是 precommitted（私有 witness）
//! - `q` 是常量（这里选用 vesta::Base 的模数，使得 q < p，其中 p 是约束域 vesta::Scalar 的模数）
//! - 电路约束当且仅当 `k < q` 时可满足

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::PrimeField;
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::Num as NumTrait;
use spartan2::bellpepper::r1cs::SpartanShape;
use spartan2::bellpepper::shape_cs::ShapeCS;
use spartan2::gadgets::{less_than::lt_constant, nonnative::nat_to_f};
use spartan2::provider::{pasta::{pallas, vesta}, PallasHyraxEngine};
use spartan2::split_zk::SpartanPrepZkSNARK;
use spartan2::split_zk::SpartanZkSNARK;
use spartan2::traits::{circuit::SpartanCircuit, snark::R1CSSNARKTrait, Engine};
use std::marker::PhantomData;
use tracing::info;
use tracing_subscriber::EnvFilter;

type E = PallasHyraxEngine;
type F = <E as Engine>::Scalar; // pallas::Scalar == Fq

fn bigint_from_biguint(x: &BigUint) -> BigInt {
  BigInt::from_biguint(Sign::Plus, x.clone())
}

fn constant_q_biguint() -> BigUint {
  // deprecated: keep for reference
  let hex = vesta::Base::MODULUS.trim_start_matches("0x");
  BigUint::from_str_radix(hex, 16).expect("invalid modulus hex")
}

fn constant_p_biguint() -> BigUint {
  // pallas::Scalar is the constraint field modulus p for this example.
  let hex = pallas::Scalar::MODULUS.trim_start_matches("0x");
  BigUint::from_str_radix(hex, 16).expect("invalid modulus hex")
}

fn choose_q_less_than_p(p: &BigUint) -> (&'static str, BigUint) {
  let candidates: [(&'static str, &str); 3] = [
    ("pallas::Base (Fp)", pallas::Base::MODULUS),
    ("vesta::Scalar (Fp)", vesta::Scalar::MODULUS),
    ("vesta::Base (Fq)", vesta::Base::MODULUS),
  ];
  for (name, modulus_hex) in candidates {
    let q = BigUint::from_str_radix(modulus_hex.trim_start_matches("0x"), 16)
      .expect("invalid modulus hex");
    if &q < p {
      return (name, q);
    }
  }
  panic!("no candidate q < p found")
}

#[derive(Clone, Debug)]
struct LtQCircuit<Scalar: PrimeField> {
  k_nat: BigUint, // integer witness for k
  _p: PhantomData<Scalar>,
}

impl<Scalar: PrimeField> LtQCircuit<Scalar> {
  fn new(k_nat: BigUint) -> Self {
    Self {
      k_nat,
      _p: PhantomData,
    }
  }
}

impl<Ec: Engine> SpartanCircuit<Ec> for LtQCircuit<Ec::Scalar> {
  fn public_values(&self) -> Result<Vec<Ec::Scalar>, SynthesisError> {
    // No public inputs in this demo.
    Ok(vec![])
  }

  fn shared<CS: ConstraintSystem<Ec::Scalar>>(
    &self,
    _cs: &mut CS,
  ) -> Result<Vec<AllocatedNum<Ec::Scalar>>, SynthesisError> {
    Ok(vec![])
  }

  fn precommitted<CS: ConstraintSystem<Ec::Scalar>>(
    &self,
    cs: &mut CS,
    _shared: &[AllocatedNum<Ec::Scalar>],
  ) -> Result<Vec<AllocatedNum<Ec::Scalar>>, SynthesisError> {
    let k_value: Ec::Scalar =
      nat_to_f(&bigint_from_biguint(&self.k_nat)).expect("k must fit the constraint field");
    let k = AllocatedNum::alloc(cs.namespace(|| "k"), || Ok(k_value))?;
    Ok(vec![k])
  }

  fn num_challenges(&self) -> usize {
    0
  }

  fn synthesize<CS: ConstraintSystem<Ec::Scalar>>(
    &self,
    cs: &mut CS,
    _shared: &[AllocatedNum<Ec::Scalar>],
    precommitted: &[AllocatedNum<Ec::Scalar>],
    _challenges: Option<&[Ec::Scalar]>,
  ) -> Result<(), SynthesisError> {
    let k = &precommitted[0];
    let q = constant_q_biguint();

    // lt = 1 iff k < q
    let lt = lt_constant(cs.namespace(|| "k < q"), k, &q)?;

    // Enforce lt == 1
    cs.enforce(
      || "require lt == 1",
      |lc| lc + lt.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + CS::one(),
    );
    Ok(())
  }
}

fn main() {
  tracing_subscriber::fmt()
    .with_target(false)
    .with_ansi(true)
    .with_env_filter(EnvFilter::from_default_env())
    .init();

  let p = constant_p_biguint();
  let (q_src, q) = choose_q_less_than_p(&p);
  info!("Using q from {q_src}; q < p = {}", q < p);

  // Satisfying witness: k = q - 1
  let k_nat = &q - BigUint::from(1u64);
  let circuit = LtQCircuit::<F>::new(k_nat);

  // Show constraint count (unpadded)
  let shape =
    <ShapeCS<E> as SpartanShape<E>>::r1cs_shape(&circuit).expect("failed to build R1CS shape");
  let sizes = shape.sizes();
  info!("LtQ circuit: num_constraints_unpadded = {}", sizes[0]);

  // Full SNARK flow
  let (pk, vk) = SpartanZkSNARK::<E>::setup(circuit.clone()).expect("setup failed");
  let prep: SpartanPrepZkSNARK<E> =
    SpartanZkSNARK::<E>::prep_prove(&pk, circuit.clone(), false).expect("prep_prove failed");
  let proof = SpartanZkSNARK::<E>::prove(&pk, circuit.clone(), &prep, false).expect("prove failed");
  proof.verify(&vk).expect("verify failed");

  info!("LtQ example completed successfully");
}

