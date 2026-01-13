// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! MAC example:
//! - 构造一个模 2^128 域上的 MAC 电路：证明 a * msg + b = mac (mod 2^128)
//! - 其中 a = a0 + a1, b = b0 + b1
//! - a0, b0, msg 是私有的，a1, b1, mac 是公开的
//! - 使用 `BigNat` 实现模 2^128 的运算，并统计该电路的 R1CS 约束数量
//! - 通过 `SpartanZkSNARK` 完整跑一遍 {setup, prove, verify} 流程

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

/// MAC 电路：
/// 语句：公开 a1, b1, mac，证明存在私有 a0, b0, msg，使得 (a0 + a1) * msg + (b0 + b1) = mac (mod 2^128)
#[derive(Clone, Debug)]
struct MacCircuit<Scalar: ff::PrimeField + PrimeFieldBits> {
  /// 私有密钥 a0
  a0_value: BigInt,
  /// 私有密钥 b0
  b0_value: BigInt,
  /// 私有消息 msg
  msg_value: BigInt,
  /// 公开密钥 a1
  a1_value: BigInt,
  /// 公开密钥 b1
  b1_value: BigInt,
  /// 公开 MAC 值
  mac_value: BigInt,
  _p: PhantomData<Scalar>,
}

impl<Scalar: ff::PrimeField + PrimeFieldBits> MacCircuit<Scalar> {
  fn new() -> Self {
    use rand_core::{OsRng, RngCore};
    let mut rng = OsRng;
    let a0_value = BigInt::from((rng.next_u64() as u128) << 64 | rng.next_u64() as u128);
    let b0_value = BigInt::from((rng.next_u64() as u128) << 64 | rng.next_u64() as u128);
    let msg_value = BigInt::from((rng.next_u64() as u128) << 64 | rng.next_u64() as u128);
    let a1_value = BigInt::from((rng.next_u64() as u128) << 64 | rng.next_u64() as u128);
    let b1_value = BigInt::from((rng.next_u64() as u128) << 64 | rng.next_u64() as u128);
    let mac_value = ((&a0_value + &a1_value) * &msg_value + &b0_value + &b1_value) % modulus_2_128();
    Self {
      a0_value,
      b0_value,
      msg_value,
      a1_value,
      b1_value,
      mac_value,
      _p: PhantomData,
    }
  }
}

impl<Ec: Engine> SpartanCircuit<Ec> for MacCircuit<Ec::Scalar> {
  fn public_values(&self) -> Result<Vec<Ec::Scalar>, SynthesisError> {
    // 公开 a1, b1, mac（都是公开的）    
    Ok(vec![nat_to_f(&self.a1_value).unwrap(), nat_to_f(&self.b1_value).unwrap(), nat_to_f(&self.mac_value).unwrap()])
  }

  fn shared<CS: ConstraintSystem<Ec::Scalar>>(
    &self,
    cs: &mut CS,
  ) -> Result<Vec<AllocatedNum<Ec::Scalar>>, SynthesisError> {
    let msg = AllocatedNum::alloc(cs.namespace(|| "msg"), || {
      Ok(nat_to_f(&self.msg_value).unwrap())
    })?;
    Ok(vec![msg])
  }

  fn precommitted<CS: ConstraintSystem<Ec::Scalar>>(
    &self,
    cs: &mut CS,
    _shared: &[AllocatedNum<Ec::Scalar>],
  ) -> Result<Vec<AllocatedNum<Ec::Scalar>>, SynthesisError> {
    let a0 = AllocatedNum::alloc(cs.namespace(|| "a0"), || {
      Ok(nat_to_f(&self.a0_value).unwrap())
    })?;
    let b0 = AllocatedNum::alloc(cs.namespace(|| "b0"), || {
      Ok(nat_to_f(&self.b0_value).unwrap())
    })?;
    Ok(vec![a0, b0])
  }

  fn num_challenges(&self) -> usize {
    // Mac circuit expects 2 challenges: a1, b1
    0 // a1, b1
  }

  fn synthesize<CS: ConstraintSystem<Ec::Scalar>>(
    &self,
    cs: &mut CS,
    shared: &[AllocatedNum<Ec::Scalar>],
    precommitted: &[AllocatedNum<Ec::Scalar>],
    _challenges: Option<&[Ec::Scalar]>,
  ) -> Result<(), SynthesisError> {
    // 1. 创建模 2^128 的常量
    let modulus = BigNat::alloc_from_nat(
      cs.namespace(|| "modulus_2_128"),
      || Ok(modulus_2_128()),
      LIMB_WIDTH,
      N_LIMBS,
    )?;

    // 2. 将 precommitted 和 shared 变量转换为 BigNat
    let a0_num = Num::new(
      precommitted[0].get_value().as_ref().copied(),
      LinearCombination::zero() + precommitted[0].get_variable(),
    );
    let a0 = BigNat::from_num(
      cs.namespace(|| "a0_bignat"),
      &a0_num,
      LIMB_WIDTH,
      N_LIMBS,
    )?;
    a0.assert_well_formed(cs.namespace(|| "a0 well-formed"))?;

    let b0_num = Num::new(
      precommitted[1].get_value().as_ref().copied(),
      LinearCombination::zero() + precommitted[1].get_variable(),
    );
    let b0 = BigNat::from_num(
      cs.namespace(|| "b0_bignat"),
      &b0_num,
      LIMB_WIDTH,
      N_LIMBS,
    )?;
    b0.assert_well_formed(cs.namespace(|| "b0 well-formed"))?;

    let msg_num = Num::new(
      shared[0].get_value().as_ref().copied(),
      LinearCombination::zero() + shared[0].get_variable(),
    );
    let msg = BigNat::from_num(
      cs.namespace(|| "msg_bignat"),
      &msg_num,
      LIMB_WIDTH,
      N_LIMBS,
    )?;
    msg.assert_well_formed(cs.namespace(|| "msg well-formed"))?;

    // 3. 分配公开的 a1, b1, mac（作为公开输入）
    // 注意：public_values 返回的顺序是 [a1, b1, mac]
    // 我们需要按照这个顺序从公开输入中读取，然后转换为 BigNat
    
    // 读取 a1（第一个公开输入）并转换为 BigNat
    let a1_input = AllocatedNum::alloc_input(cs.namespace(|| "a1_input"), || {
      Ok(nat_to_f(&self.a1_value).unwrap())
    })?;
    // 使用 BigNat::from_num 将 AllocatedNum 转换为 BigNat
    let a1_num = Num::new(
      a1_input.get_value().as_ref().copied(),
      LinearCombination::zero() + a1_input.get_variable(),
    );
    let a1 = BigNat::from_num(
      cs.namespace(|| "a1"),
      &a1_num,
      LIMB_WIDTH,
      N_LIMBS,
    )?;
    a1.assert_well_formed(cs.namespace(|| "a1 well-formed"))?;
    
    // 读取 b1（第二个公开输入）并转换为 BigNat
    let b1_input = AllocatedNum::alloc_input(cs.namespace(|| "b1_input"), || {
      Ok(nat_to_f(&self.b1_value).unwrap())
    })?;
    let b1_num = Num::new(
      b1_input.get_value().as_ref().copied(),
      LinearCombination::zero() + b1_input.get_variable(),
    );
    let b1 = BigNat::from_num(
      cs.namespace(|| "b1"),
      &b1_num,
      LIMB_WIDTH,
      N_LIMBS,
    )?;
    b1.assert_well_formed(cs.namespace(|| "b1 well-formed"))?;
    // 读取 mac（第三个公开输入）并转换为 BigNat
    let mac_input = AllocatedNum::alloc_input(cs.namespace(|| "mac_input"), || {
      Ok(nat_to_f(&self.mac_value).unwrap())
    })?;
    let mac_num = Num::new(
      mac_input.get_value().as_ref().copied(),
      LinearCombination::zero() + mac_input.get_variable(),
    );
    let mac_public = BigNat::from_num(
      cs.namespace(|| "mac_public"),
      &mac_num,
      LIMB_WIDTH,
      N_LIMBS,
    )?;
    mac_public.assert_well_formed(cs.namespace(|| "mac_public well-formed"))?;
    
    // 5. 计算 a = a0 + a1
    let a = a0.add(&a1)?;
    let a = a.red_mod(cs.namespace(|| "a mod 2^128"), &modulus)?;

    // 6. 计算 b = b0 + b1
    let b = b0.add(&b1)?;
    let b = b.red_mod(cs.namespace(|| "b mod 2^128"), &modulus)?;

    // 7. 计算 a * msg (mod 2^128)
    let (_, a_times_msg) = a.mult_mod(cs.namespace(|| "a * msg mod 2^128"), &msg, &modulus)?;
    // mult_mod 已经返回 remainder，不需要再次 red_mod 
    // 8. 计算 (a * msg) + b (mod 2^128)
    let sum = a_times_msg.add(&b)?;
    let computed_mac = sum.red_mod(cs.namespace(|| "(a*msg+b) mod 2^128"), &modulus)?;
    computed_mac.assert_well_formed(cs.namespace(|| "computed_mac well-formed"))?;

    // 9. 约束 computed_mac == mac_public
    computed_mac.equal_when_carried_regroup(cs.namespace(|| "mac equality"), &mac_public)?;

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
  
  let circuit = MacCircuit::<<E as Engine>::Scalar>::new();

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
  // MAC 电路的 witness 是 128 位整数，不适合 is_small=true 的优化
  let (pk, vk) = SpartanZkSNARK::<E>::setup(circuit.clone()).expect("setup failed");
  let prep = SpartanZkSNARK::<E>::prep_prove(&pk, circuit.clone(), false).expect("prep_prove failed");
  let proof = SpartanZkSNARK::<E>::prove(&pk, circuit.clone(), &prep, false).expect("prove failed");
  proof.verify(&vk).expect("verify failed");

  info!("MAC example (mod 2^128) completed successfully");
}
