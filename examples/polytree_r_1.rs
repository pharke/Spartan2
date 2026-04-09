// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Relation \Rel_{1}（ECC + mod-p 示例）
//!
//! 公开输入：c, delta[2], K(x,y), C1(x,y)
//! precommitted：x', o2, mu[2], V(x,y)
//!
//! 约束（当前实现）：
//! - V + [o2]K = C1
//! - x(V) = x'（等价于 x - x' = kq 且 x',x<q 时 k=0 的情形）
//! - delta_0 = c*(mu_0 + x') mod p
//! - delta_1 = c*(mu_1 + o2) mod p

use bellpepper::gadgets::boolean::field_into_allocated_bits_le;
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::{Field, PrimeField, PrimeFieldBits};
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::Num as NumTrait;
use spartan2::bellpepper::r1cs::SpartanShape;
use spartan2::bellpepper::shape_cs::ShapeCS;
use spartan2::split_zk::SpartanPrepZkSNARK;
use spartan2::{
  gadgets::{
    ecc::AllocatedPoint,
    nonnative::nat_to_f,
  },
  provider::{pasta::vesta, PallasHyraxEngine},
  provider::traits::DlogGroup,
  split_zk::SpartanZkSNARK,
  traits::{circuit::SpartanCircuit, snark::R1CSSNARKTrait, Engine},
};
use std::marker::PhantomData;
use tracing::info;
use tracing_subscriber::EnvFilter;

type E = PallasHyraxEngine;
type RelField = <E as Engine>::Scalar; // pallas::Scalar == vesta::Base (Fq)
type CurveG = vesta::Point;
type PModField = RelField; // delta works in relation field (mod p)

fn modulus_p() -> BigInt {
  let modules_uint = BigUint::from_str_radix(&PModField::MODULUS[2..], 16).unwrap();
  BigInt::from_biguint(Sign::Plus, modules_uint)
}

fn rel_to_nat(x: RelField) -> BigInt {
  BigInt::from_bytes_le(Sign::Plus, x.to_repr().as_ref())
}

#[derive(Clone, Debug)]
struct PolyTreeR1Circuit<Scalar: ff::PrimeField + PrimeFieldBits> {
  c: Scalar,
  delta: [Scalar; 2],
  c1_xy: (Scalar, Scalar),
  xp: Scalar,
  o2: Scalar,
  mu: [Scalar; 2],
  v_xy: (Scalar, Scalar),
  _p: PhantomData<Scalar>,
}

impl SpartanCircuit<E> for PolyTreeR1Circuit<RelField> {
  fn public_values(&self) -> Result<Vec<RelField>, SynthesisError> {
    Ok(vec![
      self.c,
      self.delta[0],
      self.delta[1],
      self.c1_xy.0,
      self.c1_xy.1,
    ])
  }

  fn shared<CS: ConstraintSystem<RelField>>(
    &self,
    _cs: &mut CS,
  ) -> Result<Vec<AllocatedNum<RelField>>, SynthesisError> {
    Ok(vec![])
  }

  fn precommitted<CS: ConstraintSystem<RelField>>(
    &self,
    cs: &mut CS,
    _shared: &[AllocatedNum<RelField>],
  ) -> Result<Vec<AllocatedNum<RelField>>, SynthesisError> {
    let xp = AllocatedNum::alloc(cs.namespace(|| "xp"), || Ok(self.xp))?;
    let o2 = AllocatedNum::alloc(cs.namespace(|| "o2"), || Ok(self.o2))?;
    let mu0 = AllocatedNum::alloc(cs.namespace(|| "mu0"), || Ok(self.mu[0]))?;
    let mu1 = AllocatedNum::alloc(cs.namespace(|| "mu1"), || Ok(self.mu[1]))?;
    Ok(vec![xp, o2, mu0, mu1])
  }

  fn num_challenges(&self) -> usize {
    0
  }

  fn synthesize<CS: ConstraintSystem<RelField>>(
    &self,
    cs: &mut CS,
    _shared: &[AllocatedNum<RelField>],
    precommitted: &[AllocatedNum<RelField>],
    _challenges: Option<&[RelField]>,
  ) -> Result<(), SynthesisError> {
    let c_input = AllocatedNum::alloc_input(cs.namespace(|| "c"), || Ok(self.c))?;
    let delta0_pub = AllocatedNum::alloc_input(cs.namespace(|| "delta0"), || Ok(self.delta[0]))?;
    let delta1_pub = AllocatedNum::alloc_input(cs.namespace(|| "delta1"), || Ok(self.delta[1]))?;
    let c1x_pub = AllocatedNum::alloc_input(cs.namespace(|| "C1x"), || Ok(self.c1_xy.0))?;
    let c1y_pub = AllocatedNum::alloc_input(cs.namespace(|| "C1y"), || Ok(self.c1_xy.1))?;

    let xp = precommitted[0].clone();
    let o2 = precommitted[1].clone();
    let mu0 = precommitted[2].clone();
    let mu1 = precommitted[3].clone();

    // K is a fixed constant base point.
    let k_coords = CurveG::generator().to_coordinates();
    let k = AllocatedPoint::<E, CurveG>::alloc(
      cs.namespace(|| "K_const"),
      Some((k_coords.0, k_coords.1, false)),
    )?;
    // k.check_on_curve(cs.namespace(|| "K on curve"))?;

    let c1 = AllocatedPoint::<E, CurveG>::alloc(
      cs.namespace(|| "C1_public_point"),
      Some((self.c1_xy.0, self.c1_xy.1, false)),
    )?;
    c1.check_on_curve(cs.namespace(|| "C1 on curve"))?;
    cs.enforce(|| "C1x bind", |lc| lc + c1.x.get_variable() - c1x_pub.get_variable(), |lc| lc + CS::one(), |lc| lc);
    cs.enforce(|| "C1y bind", |lc| lc + c1.y.get_variable() - c1y_pub.get_variable(), |lc| lc + CS::one(), |lc| lc);

    let o2_bits = field_into_allocated_bits_le(cs.namespace(|| "o2_bits"), o2.get_value().as_ref().copied())?;
    let o2k = k.scalar_mul(cs.namespace(|| "[o2]K"), &o2_bits)?;
    let v = AllocatedPoint::<E, CurveG>::alloc(
      cs.namespace(|| "V_witness"),
      Some((self.v_xy.0, self.v_xy.1, false)),
    )?;
    v.check_on_curve(cs.namespace(|| "V on curve"))?;
    let c1_check = v.add(cs.namespace(|| "C1 = V + [o2]K"), &o2k)?;
    cs.enforce(|| "C1x relation", |lc| lc + c1_check.x.get_variable() - c1.x.get_variable(), |lc| lc + CS::one(), |lc| lc);
    cs.enforce(|| "C1y relation", |lc| lc + c1_check.y.get_variable() - c1.y.get_variable(), |lc| lc + CS::one(), |lc| lc);

    // x - x' = 0 (the k=0 case of x - x' = kq with x,x' in field range)
    cs.enforce(|| "x equals xp", |lc| lc + v.x.get_variable() - xp.get_variable(), |lc| lc + CS::one(), |lc| lc);

    let delta_term0 = mu0.add(cs.namespace(|| "mu0+xp"), &xp)?;
    let delta0 = c_input.mul(cs.namespace(|| "delta0_calc"), &delta_term0)?;
    cs.enforce(|| "delta0 eq", |lc| lc + delta0.get_variable() - delta0_pub.get_variable(), |lc| lc + CS::one(), |lc| lc);

    let delta_term1 = mu1.add(cs.namespace(|| "mu1+o2"), &o2)?;
    let delta1 = c_input.mul(cs.namespace(|| "delta1_calc"), &delta_term1)?;
    cs.enforce(|| "delta1 eq", |lc| lc + delta1.get_variable() - delta1_pub.get_variable(), |lc| lc + CS::one(), |lc| lc);
    Ok(())
  }
}

fn main() {
  tracing_subscriber::fmt()
    .with_target(false)
    .with_ansi(true)
    .with_env_filter(EnvFilter::from_default_env())
    .init();

  use rand_core::OsRng;
  let mut rng = OsRng;

  let c = RelField::random(&mut rng);
  let o2 = RelField::from(5u64); // keep scalar small for demo
  let mu0 = RelField::random(&mut rng);
  let mu1 = RelField::random(&mut rng);

  let k_host = CurveG::generator();
  // Native EC uses Vesta's scalar field (Fp); coordinates still live in vesta::Base = RelField (Fq).
  let o2_native = vesta::Scalar::from(5u64);
  let o2k_host = k_host * o2_native;
  let v_host = CurveG::generator() * vesta::Scalar::from(7u64);
  let c1_host = v_host + o2k_host;

  let v_coords = v_host.to_coordinates();
  let c1_coords = c1_host.to_coordinates();

  let c1_xy = (c1_coords.0, c1_coords.1);
  let v_xy = (v_coords.0, v_coords.1);
  let xp = v_xy.0;

  let p = modulus_p();
  let c_nat = rel_to_nat(c);
  let delta0_nat = ((&c_nat * (rel_to_nat(mu0) + rel_to_nat(xp))) % &p + &p) % &p;
  let delta1_nat = ((&c_nat * (rel_to_nat(mu1) + rel_to_nat(o2))) % &p + &p) % &p;
  let delta0 = nat_to_f::<RelField>(&delta0_nat).unwrap();
  let delta1 = nat_to_f::<RelField>(&delta1_nat).unwrap();

  let circuit = PolyTreeR1Circuit::<RelField> {
    c,
    delta: [delta0, delta1],
    c1_xy,
    xp,
    o2,
    mu: [mu0, mu1],
    v_xy,
    _p: PhantomData,
  };

  let shape = <ShapeCS<E> as SpartanShape<E>>::r1cs_shape(&circuit).expect("shape");
  info!("PolyTreeR1: num_constraints_unpadded = {}", shape.sizes()[0]);

  let (pk, vk) = SpartanZkSNARK::<E>::setup(circuit.clone()).expect("setup failed");
  let prep: SpartanPrepZkSNARK<E> =
    SpartanZkSNARK::<E>::prep_prove(&pk, circuit.clone(), false).expect("prep_prove failed");
  let proof = SpartanZkSNARK::<E>::prove(&pk, circuit.clone(), &prep, false).expect("prove failed");
  proof.verify(&vk).expect("verify failed");
  info!("PolyTreeR1 example completed successfully");
}