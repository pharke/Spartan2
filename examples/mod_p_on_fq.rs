// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! 模 p 运算电路示例
//! - 电路：证明 d = u + cx mod p
//! - u,c 在[0,min(q,p)-1]范围内
//! - x,d 在[0,p-1]范围内
//! - c,d 是公开的， u,x 是私有的，p 是模数
//! - 用于调试非q 域的模运算的基本问题

use bellpepper_core::{
    num::AllocatedNum,
    ConstraintSystem,
    LinearCombination,
    SynthesisError,
  };
  use ff::{Field, PrimeField, PrimeFieldBits};
  use num_bigint::{BigInt, BigUint, Sign};
  use num_traits::Num as NumTrait;
  use spartan2::bellpepper::r1cs::SpartanShape;
  use spartan2::bellpepper::shape_cs::ShapeCS;
  use spartan2::split_zk::SpartanPrepZkSNARK;
use spartan2::{
    gadgets::nonnative::{BigNat, Num, nat_to_f},
    provider::{pasta::pallas, PallasHyraxEngine},
    split_zk::SpartanZkSNARK,
    traits::{
      circuit::SpartanCircuit,
      snark::R1CSSNARKTrait,
      Engine,
    },
  };
  use std::marker::PhantomData;
  use tracing::info;
  use tracing_subscriber::EnvFilter;
  
  type E = PallasHyraxEngine;
  type Fq = pallas::Scalar;
  type Fp = pallas::Base;
  
  // 模 p,q 的参数（与 BigNat 分解一致）
  const LIMB_WIDTH: usize = 16; // 每个 limb 16 位
  const N_LIMBS: usize = 16; // 16 * 16 = 256 位，可覆盖 p、q

  fn modulus_p() -> BigInt {
    // the byte representation of the base field of Pallas curve (little-endian)
    let modules_uint = BigUint::from_str_radix(&Fp::MODULUS[2..], 16).unwrap();
    let modulus = BigInt::from_biguint(Sign::Plus, modules_uint);
    modulus
  }

  fn modulus_q() -> BigInt {
    // the byte representation of the scalar field of Pallas curve (little-endian)
    let modules_uint = BigUint::from_str_radix(&Fq::MODULUS[2..], 16).unwrap();
    let modulus = BigInt::from_biguint(Sign::Plus, modules_uint);
    modulus
  }


  fn fq_to_scalar<Scalar: ff::PrimeField + PrimeFieldBits>(x: Fq) -> Option<Scalar> {
    let x_bigint = fq_to_nat(x);
    nat_to_f(&x_bigint)
  }
  /// Pallas 标量域元素的标准字节表示是 **little-endian**（见 halo2curves `endian = "little"`）。
  /// 这里必须与 `nat_to_f` / 电路 witness 一致，误用 `from_bytes_be` 会得到错误的整数，
  /// 进而使链外计算的 `d` 与 `BigNat` 约束不一致，验证失败。
  fn fq_to_nat(x: Fq) -> BigInt {
    BigInt::from_bytes_le(Sign::Plus, x.to_repr().as_ref())
  }

  /// 与 `synthesize` 中 BigNat 语义一致：先 `(u + x)`，再 `(c * ·) mod p`，最后 `mod q`。
  fn d_bigint_matching_circuit(u: &BigInt, x: &BigInt, c: &BigInt) -> BigInt {
    let p = modulus_p();
    let q = modulus_q();
    let sum = u + x;
    let cux_mod_p = (c * sum) % &p;
    cux_mod_p % &q
  }
  
  /// 最简单的模 p 电路： 设 q < p,
  /// 语句：d_q \in [0,q-1],
  /// 证明存在私有precommitted的 u , x_q \in [0,q-1],
  /// 和派生的挑战 c \in [0,q-1],
  /// 使得 d_q = u + cx mod p mod q 成立
  #[derive(Clone, Debug)]
  struct ModPOnFQCircuit<Scalar: ff::PrimeField + PrimeFieldBits> {
    /// 私有值 u, x_q
    u_value_q: BigInt,
    x_value_q: BigInt,
    /// 公开值 d_q
    d_value_q: BigInt,
    _p: PhantomData<Scalar>,
  }
  
  impl<Scalar: ff::PrimeField + PrimeFieldBits> ModPOnFQCircuit<Scalar> {
    fn new(u_value_q: BigInt, x_value_q: BigInt, d_value_q: BigInt) -> Self {
        //判断q是否小于p
        if modulus_q() < modulus_p() {
          println!("q must be less than p");
        }
      Self {
        u_value_q,
        x_value_q,
        d_value_q,
        _p: PhantomData,
      }
    }
  }

  fn allocated_num_to_bignat<Scalar: ff::PrimeField + PrimeFieldBits,CS: ConstraintSystem<Scalar>>(
    cs: &mut CS,
    label: &'static str,
    x: AllocatedNum<Scalar>,
  ) -> BigNat<Scalar> {
    let x_num = Num::new(
      x.get_value().as_ref().copied(),
      LinearCombination::zero() + x.get_variable(),
    );
    BigNat::from_num(cs.namespace(|| label), &x_num, LIMB_WIDTH, N_LIMBS).unwrap()
  }
  
  impl<Ec: Engine> SpartanCircuit<Ec> for ModPOnFQCircuit<Ec::Scalar> {
    fn public_values(&self) -> Result<Vec<Ec::Scalar>, SynthesisError> {
      // 公开 c, d_h, d_l
      Ok(vec![nat_to_f(&self.d_value_q).unwrap()])
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
      let u_value_q = AllocatedNum::alloc(cs.namespace(|| "u_value_q"), || Ok(nat_to_f(&self.u_value_q).unwrap()))?;
      let x_value_q = AllocatedNum::alloc(cs.namespace(|| "x_value_q"), || Ok(nat_to_f(&self.x_value_q).unwrap()))?;
      Ok(vec![u_value_q, x_value_q])
    }
  
    fn num_challenges(&self) -> usize {
      1
    }
  
    fn synthesize<CS: ConstraintSystem<Ec::Scalar>>(
      &self,
      cs: &mut CS,
      _shared: &[AllocatedNum<Ec::Scalar>],
      precommitted: &[AllocatedNum<Ec::Scalar>],
      challenges: Option<&[Ec::Scalar]>,
    ) -> Result<(), SynthesisError> {
      // 1. 创建所有的模数常量
      let modulus_p = BigNat::alloc_from_nat(
        cs.namespace(|| "modulus_p"),
        || Ok(modulus_p()),
        LIMB_WIDTH,
        N_LIMBS,
      )?;
  
      // 2. 取出 precommitted 的值到BigNat
      let u_value_q = allocated_num_to_bignat(cs, "u_value_q_bignat", precommitted[0].clone());
      let x_value_q = allocated_num_to_bignat(cs, "x_value_q_bignat", precommitted[1].clone());

      // 3. 分配公开的 d_value_q（关系域元素）
      let d_output = AllocatedNum::alloc_input(cs.namespace(|| "d_value_q_input"), || {
        Ok(nat_to_f(&self.d_value_q).unwrap())
      })?;
      
      // 3. 读取挑战 c（第二个公开输入）并转换为 BigNat
      let c_input = AllocatedNum::alloc_input(cs.namespace(|| "c_input"), || {
        Ok(challenges.unwrap()[0])
      })?;
      let c = allocated_num_to_bignat(cs, "c_input_bignat", c_input);
      
      // 4. 改为 d = c * (u + x) mod p
      let sum_ux = u_value_q.add(&x_value_q)?;
      let (_quotient, d_value_mod_p) =
        c.mult_mod(cs.namespace(|| "c*(u+x) mod p"), &sum_ux, &modulus_p)?;

      // 5. 不再显式做整数 mod q，而是将 mod p 的结果重组成关系域元素并与公开输入比较
      //    这证明的是域同余语义：d_output == d_value_mod_p (mod q)
      let d_output_num = Num::new(
        d_output.get_value().as_ref().copied(),
        LinearCombination::zero() + d_output.get_variable(),
      );
      let d_mod_p_bits = d_value_mod_p.decompose(cs.namespace(|| "d_value_mod_p_bits"))?;
      d_output_num.is_equal(cs.namespace(|| "d_output == d_value_mod_p in Fq"), &d_mod_p_bits);
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
  
    use rand_core::{OsRng};
    let mut rng = OsRng;
  
    let u_bigint = fq_to_nat(Fq::random(&mut rng));
    let x_bigint = fq_to_nat(Fq::random(&mut rng));
    let c_fq = Fq::random(&mut rng);
    let c_scalar_vec: Vec<<E as Engine>::Scalar> =
    vec![fq_to_scalar::<<E as Engine>::Scalar>(c_fq).unwrap()];
    let c_bigint = fq_to_nat(c_fq.clone());
    let d_bigint = d_bigint_matching_circuit(&u_bigint, &x_bigint, &c_bigint);
    let circuit = ModPOnFQCircuit::<<E as Engine>::Scalar>::new(
      u_bigint.clone(),
      x_bigint.clone(),
      d_bigint.clone(),
    );
  
    
    // 先仅生成形状，统计约束数量（未 padding 前）
    let shape =
      <ShapeCS<E> as SpartanShape<E>>::r1cs_shape(&circuit).expect("failed to build R1CS shape");
    let sizes = shape.sizes();
    let num_cons_unpadded = sizes[0];
    info!(
      "ModPOnFQ circuit: num_constraints_unpadded = {}",
      num_cons_unpadded
    );
  
    // 完整跑一遍 Spartan ZK SNARK 流程
    let (pk, vk) = SpartanZkSNARK::<E>::setup(circuit.clone()).expect("setup failed");
    let mut prep: SpartanPrepZkSNARK<E> =
      SpartanZkSNARK::<E>::prep_prove(&pk, circuit.clone(), false).expect("prep_prove failed");
    prep
      .set_external_challenges(c_scalar_vec.clone())
      .expect("set_external_challenges failed");
    let proof = SpartanZkSNARK::<E>::prove(&pk, circuit.clone(), &prep, false).expect("prove failed");
    proof.verify(&vk).expect("verify failed");

    info!("ModPOnFQ example completed successfully");
  }