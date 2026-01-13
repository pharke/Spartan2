// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! 最小化的模 2^128 运算电路示例
//! - 最简单的电路：证明 a + b = c (mod 2^128)
//! - a 是私有的，b 和 c 是公开的
//! - 用于调试模 2^128 运算的基本问题

use bellpepper_core::{
  num::AllocatedNum,
  ConstraintSystem,
  LinearCombination,
  SynthesisError,
};
use ff::PrimeFieldBits;
use num_bigint::BigInt;
use spartan2::bellpepper::r1cs::SpartanShape;
use spartan2::bellpepper::shape_cs::ShapeCS;
use spartan2::{
  gadgets::nonnative::{BigNat, Num, nat_to_f},
  provider::T256HyraxEngine,
  spartan_zk::SpartanZkSNARK,
  traits::{
    circuit::SpartanCircuit,
    snark::R1CSSNARKTrait,
    Engine,
  },
};
use std::marker::PhantomData;
use tracing::info;
use tracing_subscriber::EnvFilter;

type E = T256HyraxEngine;

// 模 2^128 的参数
const LIMB_WIDTH: usize = 16; // 每个 limb 32 位
const N_LIMBS: usize = 9;      // 9 个 limb = 144 位（用于表示值）

// 计算 2^128 作为 BigInt
fn modulus_2_128() -> BigInt {
  let mut m = BigInt::from(1u128);
  m <<= 128;
  m
}

/// 最简单的模 2^128 加法电路：
/// 语句：公开 b, c，证明存在私有 a，使得 a + b = c (mod 2^128)
#[derive(Clone, Debug)]
struct Nat128AddCircuit<Scalar: ff::PrimeField + PrimeFieldBits> {
  /// 私有值 a（128 位，BigInt 表示）
  a_value: Option<BigInt>,
  /// 公开值 b（128 位，BigInt 表示）
  b_value: BigInt,
  c_value: BigInt,
  _p: PhantomData<Scalar>,
}

impl<Scalar: ff::PrimeField + PrimeFieldBits> Nat128AddCircuit<Scalar> {
  fn new(a_value: Option<BigInt>, b_value: BigInt) -> Self {
    let c_value = (a_value.as_ref().unwrap() * &b_value) % modulus_2_128();
    Self {
      a_value,
      b_value,
      c_value,
      _p: PhantomData,
    }
  }
}

impl<Ec: Engine> SpartanCircuit<Ec> for Nat128AddCircuit<Ec::Scalar> {
  fn public_values(&self) -> Result<Vec<Ec::Scalar>, SynthesisError> {
    // 公开 b 和 c
    Ok(vec![nat_to_f(&self.b_value).unwrap(), nat_to_f(&self.c_value).unwrap()])
  }

  fn shared<CS: ConstraintSystem<Ec::Scalar>>(
    &self,
    _cs: &mut CS,
  ) -> Result<Vec<AllocatedNum<Ec::Scalar>>, SynthesisError> {
    Ok(vec![])
  }

  fn precommitted<CS: ConstraintSystem<Ec::Scalar>>(
    &self,
    _cs: &mut CS,
    _shared: &[AllocatedNum<Ec::Scalar>],
  ) -> Result<Vec<AllocatedNum<Ec::Scalar>>, SynthesisError> {
    Ok(vec![])
  }

  fn num_challenges(&self) -> usize {
    0
  }

  fn synthesize<CS: ConstraintSystem<Ec::Scalar>>(
    &self,
    cs: &mut CS,
    _shared: &[AllocatedNum<Ec::Scalar>],
    _precommitted: &[AllocatedNum<Ec::Scalar>],
    _challenges: Option<&[Ec::Scalar]>,
  ) -> Result<(), SynthesisError> {
    // 1. 创建模 2^128 的常量
    let modulus = BigNat::alloc_from_nat(
      cs.namespace(|| "modulus_2_128"),
      || Ok(modulus_2_128()),
      LIMB_WIDTH,
      N_LIMBS,
    )?;

    // 2. 分配私有的 a
    let a = BigNat::alloc_from_nat(
      cs.namespace(|| "a"),
      || {
        Ok(self.a_value.clone().ok_or(SynthesisError::AssignmentMissing)?)
      },
      LIMB_WIDTH,
      N_LIMBS,
    )?;
    a.assert_well_formed(cs.namespace(|| "a well-formed"))?;
    
    // 3. 分配公开的 b 和 c（作为公开输入）
    // 注意：public_values 返回的顺序是 [b, c]
    
    // 读取 b（第一个公开输入）并转换为 BigNat
    let b_input = AllocatedNum::alloc_input(cs.namespace(|| "b_input"), || {
      Ok(nat_to_f(&self.b_value).unwrap())
    })?;
    let b_num = Num::new(
      b_input.get_value().as_ref().copied(),
      LinearCombination::zero() + b_input.get_variable(),
    );
    let b = BigNat::from_num(
      cs.namespace(|| "b"),
      &b_num,
      LIMB_WIDTH,
      N_LIMBS,
    )?;
    
    // 读取 c（第二个公开输入）并转换为 BigNat
    let c_input = AllocatedNum::alloc_input(cs.namespace(|| "c_input"), || {
      Ok(nat_to_f(&self.c_value).unwrap())
    })?;
    let c_num = Num::new(
      c_input.get_value().as_ref().copied(),
      LinearCombination::zero() + c_input.get_variable(),
    );
    let c = BigNat::from_num(
      cs.namespace(|| "c"),
      &c_num,
      LIMB_WIDTH,
      N_LIMBS,
    )?;
    
    // 3. 计算 (a * b) mod 2^128
    // mult_mod 返回 (quotient, remainder)，我们需要 remainder
    let (_quotient, remainder) = a.mult_mod(cs.namespace(|| "(a*b) mod 2^128"), &b, &modulus)?;
    
    // 4. 约束 remainder == c
    remainder.equal_when_carried_regroup(cs.namespace(|| "c equality"), &c)?;
    Ok(())
  }
}


fn main() {
  // 初始化日志
  tracing_subscriber::fmt()
    .with_target(false)
    .with_ansi(true)
    .with_env_filter(EnvFilter::from_default_env())
    .init();

  // 测试两种情况：
  // 1. 先测试 a + b < 2^128（不需要模运算）
  // 2. 然后测试 a + b >= 2^128（需要模运算）
  use rand_core::{OsRng, RngCore};
  let mut rng = OsRng;

  let a_bigint = BigInt::from((rng.next_u64() as u128) << 64 | rng.next_u64() as u128);
  let b_bigint = BigInt::from((rng.next_u64() as u128) << 64 | rng.next_u64() as u128);
  let circuit = Nat128AddCircuit::<<E as Engine>::Scalar>::new(
    Some(a_bigint.clone()),
    b_bigint.clone(),
  );

  
  // 先仅生成形状，统计约束数量（未 padding 前）
  let shape =
    <ShapeCS<E> as SpartanShape<E>>::r1cs_shape(&circuit).expect("failed to build R1CS shape");
  let sizes = shape.sizes();
  let num_cons_unpadded = sizes[0];
  info!(
    "MAC circuit (mod 2^128): num_constraints_unpadded = {}",
    num_cons_unpadded
  );

  // 完整跑一遍 Spartan ZK SNARK 流程
  let (pk, vk) = SpartanZkSNARK::<E>::setup(circuit.clone()).expect("setup failed");
  let prep = SpartanZkSNARK::<E>::prep_prove(&pk, circuit.clone(), false).expect("prep_prove failed");
  let proof = SpartanZkSNARK::<E>::prove(&pk, circuit.clone(), &prep, false).expect("prove failed");
  proof.verify(&vk).expect("verify failed");

  info!("Nat128 addition example (mod 2^128) completed successfully");
}

