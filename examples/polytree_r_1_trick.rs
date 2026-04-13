// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Relation \Rel_{1}（ECC + mod-p 示例）— **trick 版**（`polytree_r_1_trick`）
//!
//! 与 `polytree_r_1.rs` 的语义相同，仅 **`o2` 的标量乘路径**不同：
//!
//! - **不再**对 `o2` 做全长 `field_into_allocated_bits_le`（~255 bit），改为 **固定 128 个** witness 布尔位
//!   `b_0..b_127`（LSB→MSB），一条打包约束 `o2 = Σ 2^i b_i`，再一条 **`packed == precommitted o2`**。
//!   在 `p > 2^128` 下，这已蕴涵 **`o2` 作为无符号整数 `< 2^128`**，无需额外 `lt_constant`。
//! - **`const_base_pow2_mul_with_slack` 只消费这 128 位**，不完整加循环从 ~254 轮降到 **127 轮**，用于把门数压档。
//!
//! **协议假设**：本关系中的标量 `o2` 仅取 `< 2^128`；宿主侧 `vesta::Scalar` 必须与该整数一致。
//!
//! ---
//!
//! 公开输入：c, delta[2], K(x,y), C1(x,y)
//! precommitted：x', o2, mu[2], V(x,y)
//!
//! 约束：
//! - V + [o2]K = C1
//! - x(V) = x' + k·q（在关系域上），其中 q = |Fr(vesta)|，`x'` 与 r0 对齐取 **V 的 x 坐标对 q 取余**；
//!   并用 `lt_constant` 约束 `k < ⌊(p−1)/q⌋+1` 与 **`x' < q`**（q = |Fr(vesta)|，p 为关系域模数）。
//! - delta_0 = c*(mu_0 + x') mod p
//! - delta_1 = c*(mu_1 + o2) mod p
use bellpepper_core::{
  boolean::{AllocatedBit, Boolean},
  num::AllocatedNum,
  ConstraintSystem, LinearCombination, SynthesisError,
};
use ff::{Field, PrimeField, PrimeFieldBits};
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::Num as NumTrait;
use spartan2::bellpepper::r1cs::SpartanShape;
use spartan2::bellpepper::shape_cs::ShapeCS;
use spartan2::split_zk::SpartanPrepZkSNARK;
use spartan2::{
  gadgets::{
    ecc::{AllocatedPoint, AllocatedPointNonInfinity},
    less_than::lt_constant,
    nonnative::nat_to_f,
    utils::field_switch,
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
type VestaBase = vesta::Base;
type PModField = RelField; // delta works in relation field (mod p)

/// 用于 `[o2]K` 的 bit 数；固定基乘与打包均以此为长度。
const O2_BITS: usize = 128;

fn modulus_p() -> BigInt {
  let modules_uint = BigUint::from_str_radix(&PModField::MODULUS[2..], 16).unwrap();
  BigInt::from_biguint(Sign::Plus, modules_uint)
}

fn rel_to_nat(x: RelField) -> BigInt {
  BigInt::from_bytes_le(Sign::Plus, x.to_repr().as_ref())
}

fn bigint_from_biguint(x: &BigUint) -> BigInt {
  BigInt::from_biguint(Sign::Plus, x.clone())
}

/// 与 r0 一致：q = vesta::Scalar 域模数（`|Fr|`）。
fn modulus_q_biguint() -> BigUint {
  BigUint::from_str_radix(vesta::Scalar::MODULUS.trim_start_matches("0x"), 16)
    .expect("invalid vesta::Scalar modulus hex")
}

fn modulus_p_biguint() -> BigUint {
  BigUint::from_str_radix(PModField::MODULUS.trim_start_matches("0x"), 16)
    .expect("invalid RelField modulus hex")
}

/// 供 `lt_constant(k, bound)`：`k < bound` 即 `k <= floor((p−1)/q)`（p 为关系域模数）。
fn k_lt_bound_exclusive() -> BigUint {
  let p = modulus_p_biguint();
  let q = modulus_q_biguint();
  (&p - BigUint::from(1u64)) / &q + BigUint::from(1u64)
}

/// 固定基点 **常量乘**（按 `bit_i` 选择性累加 `2^i K`）：在电路内从 `k_curve` 分配 K 点，`acc` 从 K 起算，
/// 对 `bit_1…bit_{n-1}` 在**宿主侧**递推当前 `2^i K`，转换为 `(ox, oy)` 后做
/// `add_incomplete_const` + `conditionally_select`；再对 `bit_0` 做与 `scalar_mul` 相同的 slack。
///
/// `k_curve` 须为仿射点。`allocated_bits` 至少含 1 位（用于 `bit_0` slack）；长度 = 参与乘法的标量位数
///（本 trick 为 [`O2_BITS`]，LSB 在索引 0）。
fn const_base_pow2_mul_with_slack<CS: ConstraintSystem<RelField>>(
  mut cs: CS,
  k_curve: CurveG,
  allocated_bits: &[AllocatedBit],
) -> Result<AllocatedPoint<E, CurveG>, SynthesisError> {
  let (kx, ky, kinf) = k_curve.to_coordinates();
  assert!(!kinf, "K must be affine");
  let k_xy = (
    field_switch::<VestaBase, RelField>(kx),
    field_switch::<VestaBase, RelField>(ky),
  );
  let k_pt = AllocatedPoint::<E, CurveG>::alloc(
    cs.namespace(|| "K"),
    Some((k_xy.0, k_xy.1, false)),
  )?;

  let mut acc = AllocatedPointNonInfinity::<E, CurveG>::from_allocated_point(&k_pt);

  // 第 1 轮对应 bit_1，工作点为 2^1·K；先在宿主侧从 K 倍到 2K。
  let mut two_i_k = k_curve + k_curve;

  for (i, bit) in allocated_bits.iter().enumerate().skip(1) {
    let (x, y, inf) = two_i_k.to_coordinates();
    assert!(!inf, "2^{i}·K should be affine");
    let ox = field_switch::<VestaBase, RelField>(x);
    let oy = field_switch::<VestaBase, RelField>(y);

    let temp = acc.add_incomplete_const(
      cs.namespace(|| format!("add_incomplete_const_{i}")),
      ox,
      oy,
    )?;
    acc = AllocatedPointNonInfinity::conditionally_select(
      cs.namespace(|| format!("acc_iteration_{i}")),
      &temp,
      &acc,
      &Boolean::from(bit.clone()),
    )?;

    two_i_k = two_i_k + two_i_k;
  }

  let acc_pt = acc.to_allocated_point(&k_pt.is_infinity)?;
  let neg_k = k_pt.negate(cs.namespace(|| "negate_K"))?;
  let acc_minus_k = acc_pt.add(cs.namespace(|| "res_minus_K"), &neg_k)?;
  AllocatedPoint::<E, CurveG>::conditionally_select(
    cs.namespace(|| "remove_slack_bit0"),
    &acc_pt,
    &acc_minus_k,
    &Boolean::from(allocated_bits[0].clone()),
  )
}

/// 从 `o2` 的规范整数取低 `O2_BITS` 位（LSB 在索引 0）；宿主须保证高位全 0。
fn o2_le_bits_128(val: Option<RelField>) -> [bool; O2_BITS] {
  let mut out = [false; O2_BITS];
  if let Some(v) = val {
    let bu = BigUint::from_bytes_le(v.to_repr().as_ref());
    for i in 0..O2_BITS {
      out[i] = bu.bit(i as u64);
    }
  }
  out
}

/// 分配 `O2_BITS` 个布尔位，打包为 `o2_packed`，并约束 `o2_packed == o2`。
fn allocate_o2_bits_and_pack<CS: ConstraintSystem<RelField>>(
  mut cs: CS,
  o2: &AllocatedNum<RelField>,
) -> Result<Vec<AllocatedBit>, SynthesisError> {
  let bits_witness = o2_le_bits_128(o2.get_value().as_ref().copied());
  let mut bits = Vec::with_capacity(O2_BITS);
  for i in 0..O2_BITS {
    let b = AllocatedBit::alloc(
      cs.namespace(|| format!("o2_bit_{i}")),
      Some(bits_witness[i]),
    )?;
    bits.push(b);
  }

  let o2_packed = AllocatedNum::alloc(cs.namespace(|| "o2_packed_from_bits"), || {
    let v = o2.get_value().ok_or(SynthesisError::AssignmentMissing)?;
    Ok(v)
  })?;

  cs.enforce(
    || "pack o2 bits (LE)",
    |_| {
      let mut acc = LinearCombination::<RelField>::zero();
      let mut coeff = RelField::ONE;
      for b in &bits {
        acc = acc + (coeff, b.get_variable());
        coeff = coeff + coeff;
      }
      acc
    },
    |lc| lc + CS::one(),
    |lc| lc + o2_packed.get_variable(),
  );

  cs.enforce(
    || "packed o2 == precommitted o2",
    |lc| lc + o2_packed.get_variable() - o2.get_variable(),
    |lc| lc + CS::one(),
    |lc| lc,
  );

  Ok(bits)
}

#[derive(Clone, Debug)]
struct PolyTreeR1Circuit<Scalar: ff::PrimeField + PrimeFieldBits> {
  c: Scalar,
  delta: [Scalar; 2],
  c1_xy: (Scalar, Scalar),
  xp: Scalar,
  /// 整数商 k，满足 x(V) = x' + k·q（宿主侧与 xp 一致）
  k_q: Scalar,
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

    let c1 = AllocatedPoint::<E, CurveG>::alloc(
      cs.namespace(|| "C1_public_point"),
      Some((self.c1_xy.0, self.c1_xy.1, false)),
    )?;
    c1.check_on_curve(cs.namespace(|| "C1 on curve"))?;
    cs.enforce(|| "C1x bind", |lc| lc + c1.x.get_variable() - c1x_pub.get_variable(), |lc| lc + CS::one(), |lc| lc);
    cs.enforce(|| "C1y bind", |lc| lc + c1.y.get_variable() - c1y_pub.get_variable(), |lc| lc + CS::one(), |lc| lc);

    let o2_bits = allocate_o2_bits_and_pack(cs.namespace(|| "o2_bits_128"), &o2)?;
    // 固定基 K = 生成元：仅按 O2_BITS 位做不完整加 + bit_0 slack。
    let o2k = const_base_pow2_mul_with_slack(
      cs.namespace(|| "[o2]K_fixed_base"),
      CurveG::generator(),
      &o2_bits,
    )?;
    let v = AllocatedPoint::<E, CurveG>::alloc(
      cs.namespace(|| "V_witness"),
      Some((self.v_xy.0, self.v_xy.1, false)),
    )?;
    v.check_on_curve(cs.namespace(|| "V on curve"))?;
    let c1_check = v.add(cs.namespace(|| "C1 = V + [o2]K"), &o2k)?;
    cs.enforce(|| "C1x relation", |lc| lc + c1_check.x.get_variable() - c1.x.get_variable(), |lc| lc + CS::one(), |lc| lc);
    cs.enforce(|| "C1y relation", |lc| lc + c1_check.y.get_variable() - c1.y.get_variable(), |lc| lc + CS::one(), |lc| lc);

    // x(V) = x' + k·q：与 r0 对齐时 x' = x(V) mod q；并限制范围避免 F_p 回绕。
    // k·q 对固定 q 关于 k 线性，一条乘法门即可：A=k，B=q（常量），C=V.x−x'。
    let q_nat = modulus_q_biguint();
    let q_f = nat_to_f::<RelField>(&bigint_from_biguint(&q_nat)).expect("q fits RelField");

    let k_wit = AllocatedNum::alloc(cs.namespace(|| "k_quotient"), || Ok(self.k_q))?;
    cs.enforce(
      || "V.x - xp = k*q",
      |lc| lc + k_wit.get_variable(),
      |lc| lc + (q_f, CS::one()),
      |lc| lc + v.x.get_variable() - xp.get_variable(),
    );

    let k_bound = k_lt_bound_exclusive();
    let lt_k = lt_constant(cs.namespace(|| "k < bound"), &k_wit, &k_bound)?;
    cs.enforce(
      || "require k in range",
      |lc| lc + lt_k.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + CS::one(),
    );

    let lt_xp_q = lt_constant(cs.namespace(|| "xp < q"), &xp, &q_nat)?;
    cs.enforce(
      || "require xp < q",
      |lc| lc + lt_xp_q.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + CS::one(),
    );

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
  // 须 < 2^128；与 128-bit 打包及 vesta::Scalar 一致。
  let o2 = RelField::from(5u64);
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

  let c1_xy = (
    field_switch::<VestaBase, RelField>(c1_coords.0),
    field_switch::<VestaBase, RelField>(c1_coords.1),
  );
  let v_xy = (
    field_switch::<VestaBase, RelField>(v_coords.0),
    field_switch::<VestaBase, RelField>(v_coords.1),
  );

  // 与 r0 对齐：x' 为 V 的 x（Fq）在整数意义上对 |Fr(vesta)| 取余，再映入关系域。
  let q_big = modulus_q_biguint();
  let vx_nat = BigUint::from_bytes_le(v_coords.0.to_repr().as_ref());
  let xp_nat = &vx_nat % &q_big;
  let k_nat = (&vx_nat - &xp_nat) / &q_big;
  let xp = nat_to_f::<RelField>(&bigint_from_biguint(&xp_nat)).expect("xp fits RelField");
  let k_q = nat_to_f::<RelField>(&bigint_from_biguint(&k_nat)).expect("k fits RelField");

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
    k_q,
    o2,
    mu: [mu0, mu1],
    v_xy,
    _p: PhantomData,
  };

  let shape = <ShapeCS<E> as SpartanShape<E>>::r1cs_shape(&circuit).expect("shape");
  info!(
    "PolyTreeR1 trick (o2_bits={O2_BITS}): num_constraints_unpadded = {}",
    shape.sizes()[0]
  );

  let (pk, vk) = SpartanZkSNARK::<E>::setup(circuit.clone()).expect("setup failed");
  let prep: SpartanPrepZkSNARK<E> =
    SpartanZkSNARK::<E>::prep_prove(&pk, circuit.clone(), false).expect("prep_prove failed");
  let proof = SpartanZkSNARK::<E>::prove(&pk, circuit.clone(), &prep, false).expect("prove failed");
  proof.verify(&vk).expect("verify failed");
  info!("PolyTreeR1 trick example completed successfully");
}