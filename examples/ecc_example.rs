// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! ECC example:
//! - 构造一个简单的离散对数电路：给定公开点 Q，证明存在标量 k，使得 Q = [k]G，其中 G 为固定基点；
//! - 使用 `AllocatedPoint::scalar_mul` 实现标量乘，并统计该电路的 R1CS 约束数量；
//! - 通过 `SpartanZkSNARK` 完整跑一遍 {setup, prove, verify} 流程。

use bellpepper::gadgets::boolean::{AllocatedBit, field_into_allocated_bits_le};
use bellpepper_core::{
  num::AllocatedNum,
  ConstraintSystem,
  SynthesisError,
};
use ff::PrimeFieldBits;
use spartan2::{
  bellpepper::r1cs::SpartanShape,
  bellpepper::shape_cs::ShapeCS,
  gadgets::{ecc::AllocatedPoint, utils::field_switch},
  provider::{T256HyraxEngine, pt256::p256},
  provider::traits::DlogGroup,
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
use rand_core::OsRng;
use ff::Field;

type E = T256HyraxEngine;
// 使用 P256 曲线，其 Base 域等于 T256 的 Scalar 域
// 注意：p256::Point::Base = p256::Base，而 p256::Base = t256::Scalar (在 halo2curves 中)
type CurveG = p256::Point;

/// 一个简单的离散对数电路：
/// 语句：公开点 Q，证明存在私有标量 k，使得 Q = [k]G。
///
/// 为了方便，这里：
/// - G 取曲线上的某个固定常量点（宿主侧选定）；
/// - k 由 `scalar_bits` 给出（布尔 witness，最低位在前）；
/// - Q 由宿主代码根据 (G, k) 预先计算，并作为 public input 暴露；
#[derive(Clone, Debug)]
struct P256DlCircuit<Scalar: ff::PrimeField + PrimeFieldBits> {
  /// 标量 k 的二进制展开（用于为标量位分配期望值，方便统计约束）
  secret: Scalar,
  /// 基点 G.x（已在宿主侧转换到证明域）
  g_x: Scalar,
  /// 基点 G.y（已在宿主侧转换到证明域）
  g_y: Scalar,
  /// 预先计算好的 Q.x（已在宿主侧转换到证明域）
  q_x: Scalar,
  /// 预先计算好的 Q.y（已在宿主侧转换到证明域）
  q_y: Scalar,
  _p: PhantomData<Scalar>,
}

impl<Scalar: ff::PrimeField + PrimeFieldBits> P256DlCircuit<Scalar> {
  fn new(secret: Scalar) -> Self {
    
    // 生成一个随机的 P256 标量 k
    type P256Scalar = p256::Scalar;
    type P256Base = p256::Base;
    let secret_p256 = field_switch::<Scalar, P256Scalar>(secret);
    // P256 生成元及 Q = [k]G
    let g_p256 = CurveG::generator();
    let q_p256 = g_p256 * secret_p256;
    let (qx_base, qy_base, inf) = q_p256.to_coordinates();
    assert!(!inf, "Q should not be infinity");

    // 将 P256 Base 域坐标转换到证明域 E::Scalar（这里是 T256::Scalar）
    let g_coords = g_p256.to_coordinates();
    let gx_scalar: Scalar =
      field_switch::<P256Base, Scalar>(g_coords.0);
    let gy_scalar: Scalar =
      field_switch::<P256Base, Scalar>(g_coords.1);

    let qx_scalar: Scalar = field_switch::<P256Base, Scalar>(qx_base);
    let qy_scalar: Scalar = field_switch::<P256Base, Scalar>(qy_base);
    Self {
      secret,
      g_x: gx_scalar,
      g_y: gy_scalar,
      q_x: qx_scalar,
      q_y: qy_scalar,
      _p: PhantomData,
    }
  }
}

impl<Ec: Engine> SpartanCircuit<Ec> for P256DlCircuit<Ec::Scalar> {
  fn public_values(&self) -> Result<Vec<Ec::Scalar>, SynthesisError> {
    // 公开 Q = [k]G 的坐标
    Ok(vec![self.q_x, self.q_y])
  }

  fn shared<CS: ConstraintSystem<Ec::Scalar>>(
    &self,
    cs: &mut CS,
  ) -> Result<Vec<AllocatedNum<Ec::Scalar>>, SynthesisError> {
    let secret = AllocatedNum::alloc(cs.namespace(|| "secret"), || Ok(self.secret))?;
    Ok(vec![secret])
  }

  fn precommitted<CS: ConstraintSystem<Ec::Scalar>>(
    &self,
    _cs: &mut CS,
    _shared: &[AllocatedNum<Ec::Scalar>],
  ) -> Result<Vec<AllocatedNum<Ec::Scalar>>, SynthesisError> {
    // 本例不使用预提交变量
    Ok(vec![])
  }

  fn num_challenges(&self) -> usize {
    // 电路中没有随机挑战
    0
  }

  fn synthesize<CS: ConstraintSystem<Ec::Scalar>>(
    &self,
    cs: &mut CS,
    shared: &[AllocatedNum<Ec::Scalar>],
    _precommitted: &[AllocatedNum<Ec::Scalar>],
    _challenges: Option<&[Ec::Scalar]>,
  ) -> Result<(), SynthesisError> {
    // 1. 分配一个“基点” g，坐标由宿主侧指定（这里选用 P256 生成元）
    let g = AllocatedPoint::<Ec, CurveG>::alloc(
      cs.namespace(|| "G"),
      Some((self.g_x, self.g_y, false)),
    )?;
    g.check_on_curve(cs.namespace(|| "G on curve"))?;

    // 2.将secret转换为AllocatedBits
    let secret_bits = field_into_allocated_bits_le(cs.namespace(|| "secret_bits"), shared[0].get_value().as_ref().copied())?;

    // 3. 计算 r = [k]g，并做一个基础的 on-curve 检查
    let r = g.scalar_mul(cs.namespace(|| "scalar_mul"), &secret_bits)?;
    r.check_on_curve(cs.namespace(|| "R on curve"))?;

    // 4. 约束 r 的坐标等于公开输入 Q 的坐标
    // Q.x（公开输入，其值由宿主侧提供）
    let qx = AllocatedNum::alloc_input(cs.namespace(|| "Q_x"), || Ok(self.q_x))?;
    cs.enforce(
      || "R.x == Q.x",
      |lc| lc + r.x.get_variable() - qx.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc,
    );
    // Q.y（公开输入，其值由宿主侧提供）
    let qy = AllocatedNum::alloc_input(cs.namespace(|| "Q_y"), || Ok(self.q_y))?;
    cs.enforce(
      || "R.y == Q.y",
      |lc| lc + r.y.get_variable() - qy.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc,
    );

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

  let secret = <E as Engine>::Scalar::random(&mut OsRng);
  let circuit =
    P256DlCircuit::<<E as Engine>::Scalar>::new(secret);

  // 先仅生成形状，统计约束数量（未 padding 前）
  let shape =
    <ShapeCS<E> as SpartanShape<E>>::r1cs_shape(&circuit).expect("failed to build R1CS shape");
  let sizes = shape.sizes();
  let num_cons_unpadded = sizes[0];
  info!(
    "ECC scalar_mul circuit: num_constraints_unpadded = {} (含标量乘及基础 on-curve 约束)",
    num_cons_unpadded
  );

  // 完整跑一遍 Spartan ZK SNARK 流程
  // 注意：ECC 电路的 witness 是椭圆曲线点坐标（大整数），不适合 is_small=true 的优化
  // SHA-256 电路的 witness 主要是位（0/1），可以用 is_small=true
  let (pk, vk) = SpartanZkSNARK::<E>::setup(circuit.clone()).expect("setup failed");
  let prep = SpartanZkSNARK::<E>::prep_prove(&pk, circuit.clone(), false).expect("prep_prove failed");
  let proof = SpartanZkSNARK::<E>::prove(&pk, circuit.clone(), &prep, false).expect("prove failed");
  proof.verify(&vk).expect("verify failed");

  info!("ECC discrete-log-style example with scalar_mul completed successfully");
}

