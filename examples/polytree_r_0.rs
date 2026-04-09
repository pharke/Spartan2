// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Relation \Rel_{0} (多变量模运算电路示例)
//!
//! 公开输入：gamma, sigma[9], delta[2]
//! 外部挑战：c
//! precommitted：b, d, t, x', mu[2], o[3], rho[9]
//!
//! 约束：
//! - e = b mod p, s = e + gamma * x'
//! - rho'_0..rho'_8 按注释中的乘法模板构造
//! - sigma_i = rho_i + c * rho'_i (in Fq)
//! - mu'_0 = x', mu'_1 = o_2
//! - delta_i = c * (mu_i + mu'_i) mod p, 再投影到 Fq 与公开输入比较

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, LinearCombination, SynthesisError};
use ff::{Field, PrimeField, PrimeFieldBits};
use num_bigint::{BigInt, BigUint, Sign};
use num_traits::Num as NumTrait;
use spartan2::bellpepper::r1cs::SpartanShape;
use spartan2::bellpepper::shape_cs::ShapeCS;
use spartan2::split_zk::SpartanPrepZkSNARK;
use spartan2::{
  gadgets::{less_than::lt_constant, nonnative::{nat_to_f, BigNat, Num}},
  provider::{pasta::vesta, VestaHyraxEngine},
  split_zk::SpartanZkSNARK,
  traits::{circuit::SpartanCircuit, snark::R1CSSNARKTrait, Engine},
};
use std::marker::PhantomData;
use tracing::info;
use tracing_subscriber::EnvFilter;

type E = VestaHyraxEngine;
type Fq = vesta::Scalar;
type Fp = vesta::Base;

const LIMB_WIDTH: usize = 16;
const N_LIMBS: usize = 16;

fn modulus_p() -> BigInt {
  let modules_uint = BigUint::from_str_radix(&Fp::MODULUS[2..], 16).unwrap();
  BigInt::from_biguint(Sign::Plus, modules_uint)
}

fn modulus_q() -> BigInt {
  let modules_uint = BigUint::from_str_radix(&Fq::MODULUS[2..], 16).unwrap();
  BigInt::from_biguint(Sign::Plus, modules_uint)
}

fn modulus_q_biguint() -> BigUint {
  BigUint::from_str_radix(&Fq::MODULUS[2..], 16).unwrap()
}


fn fq_to_nat(x: Fq) -> BigInt {
  BigInt::from_bytes_le(Sign::Plus, x.to_repr().as_ref())
}

fn fq_to_scalar<Scalar: ff::PrimeField + PrimeFieldBits>(x: Fq) -> Option<Scalar> {
  nat_to_f(&fq_to_nat(x))
}

fn modn(x: &BigInt, n: &BigInt) -> BigInt {
  ((x % n) + n) % n
}

fn allocated_num_to_bignat<Scalar: ff::PrimeField + PrimeFieldBits, CS: ConstraintSystem<Scalar>>(
  cs: &mut CS,
  label: &str,
  x: AllocatedNum<Scalar>,
) -> BigNat<Scalar> {
  let x_num = Num::new(
    x.get_value().as_ref().copied(),
    LinearCombination::zero() + x.get_variable(),
  );
  BigNat::from_num(cs.namespace(|| label), &x_num, LIMB_WIDTH, N_LIMBS).unwrap()
}

fn bignat_to_allocated_num<Scalar: ff::PrimeField + PrimeFieldBits, CS: ConstraintSystem<Scalar>>(
  cs: &mut CS,
  label: &str,
  x: &BigNat<Scalar>,
) -> Result<AllocatedNum<Scalar>, SynthesisError> {
  let x_alloc = AllocatedNum::alloc(cs.namespace(|| format!("{label}_num")), || {
    let val = x.value.as_ref().ok_or(SynthesisError::AssignmentMissing)?;
    nat_to_f(val).ok_or(SynthesisError::AssignmentMissing)
  })?;
  let x_num = Num::new(
    x_alloc.get_value().as_ref().copied(),
    LinearCombination::zero() + x_alloc.get_variable(),
  );
  let bits = x.decompose(cs.namespace(|| format!("{label}_bits")))?;
  x_num.is_equal(cs.namespace(|| format!("{label}_eq")), &bits);
  Ok(x_alloc)
}

#[derive(Clone, Debug)]
struct PolyTreeR0Circuit<Scalar: ff::PrimeField + PrimeFieldBits> {
  m: usize,
  gamma: BigInt,
  sigma: [BigInt; 9],
  delta: [BigInt; 2],
  b: BigInt,
  d: BigInt,
  t: BigInt,
  xp: BigInt,
  mu: [BigInt; 2],
  o: [BigInt; 3],
  rho: [BigInt; 9],
  _p: PhantomData<Scalar>,
}

impl<Scalar: ff::PrimeField + PrimeFieldBits> SpartanCircuit<E> for PolyTreeR0Circuit<Scalar> {
  fn public_values(&self) -> Result<Vec<<E as Engine>::Scalar>, SynthesisError> {
    let mut out = Vec::with_capacity(12);
    out.push(nat_to_f(&self.gamma).unwrap());
    for i in 0..9 {
      out.push(nat_to_f(&self.sigma[i]).unwrap());
    }
    for i in 0..2 {
      out.push(nat_to_f(&self.delta[i]).unwrap());
    }
    Ok(out)
  }

  fn shared<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    &self,
    _cs: &mut CS,
  ) -> Result<Vec<AllocatedNum<<E as Engine>::Scalar>>, SynthesisError> {
    Ok(vec![])
  }

  fn precommitted<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    &self,
    cs: &mut CS,
    _shared: &[AllocatedNum<<E as Engine>::Scalar>],
  ) -> Result<Vec<AllocatedNum<<E as Engine>::Scalar>>, SynthesisError> {
    let mut vals = Vec::new();
    let push = |cs: &mut CS, name: &str, v: &BigInt| -> Result<AllocatedNum<<E as Engine>::Scalar>, SynthesisError> {
      AllocatedNum::alloc(cs.namespace(|| name), || Ok(nat_to_f(v).unwrap()))
    };
    vals.push(push(cs, "b", &self.b)?);
    vals.push(push(cs, "d", &self.d)?);
    vals.push(push(cs, "t", &self.t)?);
    vals.push(push(cs, "xp", &self.xp)?);
    for i in 0..2 {
      vals.push(push(cs, &format!("mu_{i}"), &self.mu[i])?);
    }
    for i in 0..3 {
      vals.push(push(cs, &format!("o_{i}"), &self.o[i])?);
    }
    for i in 0..9 {
      vals.push(push(cs, &format!("rho_{i}"), &self.rho[i])?);
    }
    Ok(vals)
  }

  fn num_challenges(&self) -> usize {
    1
  }

  fn synthesize<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    &self,
    cs: &mut CS,
    _shared: &[AllocatedNum<<E as Engine>::Scalar>],
    precommitted: &[AllocatedNum<<E as Engine>::Scalar>],
    challenges: Option<&[<E as Engine>::Scalar]>,
  ) -> Result<(), SynthesisError> {
    let modulus_p = BigNat::alloc_from_nat(
      cs.namespace(|| "modulus_p"),
      || Ok(modulus_p()),
      LIMB_WIDTH,
      N_LIMBS,
    )?;

    let gamma = AllocatedNum::alloc_input(cs.namespace(|| "gamma"), || Ok(nat_to_f(&self.gamma).unwrap()))?;
    let mut sigma_pub = Vec::new();
    let mut delta_pub = Vec::new();
    for i in 0..9 {
      sigma_pub.push(AllocatedNum::alloc_input(
        cs.namespace(|| format!("sigma_{i}")),
        || Ok(nat_to_f(&self.sigma[i]).unwrap()),
      )?);
    }
    for i in 0..2 {
      delta_pub.push(AllocatedNum::alloc_input(
        cs.namespace(|| format!("delta_{i}")),
        || Ok(nat_to_f(&self.delta[i]).unwrap()),
      )?);
    }

    let c_input = AllocatedNum::alloc_input(cs.namespace(|| "c_input"), || Ok(challenges.unwrap()[0]))?;

    let b = precommitted[0].clone();
    let d = precommitted[1].clone();
    let t = precommitted[2].clone();
    let xp = precommitted[3].clone();
    let mu = [precommitted[4].clone(), precommitted[5].clone()];
    let o = [precommitted[6].clone(), precommitted[7].clone(), precommitted[8].clone()];
    let rho: Vec<_> = (0..9).map(|i| precommitted[9 + i].clone()).collect();

    // e = b mod m 通过商-余数形式约束：
    // b - e = k*m (等价于 b = e + k*m), e < m, k < q/m
    // 这里取固定公开参数 m = pallas::Base::MODULUS (常量)
    let m_nat = BigUint::from(self.m.clone());
    let q_nat = modulus_q_biguint();
    let k_bound = &q_nat / &m_nat;
    let b_nat = BigUint::from_bytes_le(self.b.to_bytes_le().1.as_slice());
    let e_nat = &b_nat % &m_nat;
    let k_nat = &b_nat / &m_nat;

    let e = AllocatedNum::alloc(cs.namespace(|| "e"), || {
      nat_to_f(&BigInt::from_biguint(Sign::Plus, e_nat.clone())).ok_or(SynthesisError::AssignmentMissing)
    })?;
    let k = AllocatedNum::alloc(cs.namespace(|| "k"), || {
      nat_to_f(&BigInt::from_biguint(Sign::Plus, k_nat.clone())).ok_or(SynthesisError::AssignmentMissing)
    })?;
    let m_f = nat_to_f::<Fq>(&BigInt::from_biguint(Sign::Plus, m_nat.clone())).unwrap();
    cs.enforce(
      || "b = e + k*m",
      |lc| lc + k.get_variable(),
      |lc| lc + (m_f, CS::one()),
      |lc| lc + b.get_variable() - e.get_variable(),
    );
    let e_lt_m = lt_constant(cs.namespace(|| "e < m"), &e, &m_nat)?;
    cs.enforce(
      || "enforce e < m",
      |lc| lc + e_lt_m.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + CS::one(),
    );
    let k_lt_q_div_m = lt_constant(cs.namespace(|| "k < q_div_m"), &k, &k_bound)?;
    cs.enforce(
      || "enforce k < q_div_m",
      |lc| lc + k_lt_q_div_m.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + CS::one(),
    );

    // s = e + gamma * x'
    let gamma_xp = gamma.mul(cs.namespace(|| "gamma*xp"), &xp)?;
    let s = e.add(cs.namespace(|| "s=e+gamma*xp"), &gamma_xp)?;

    // rho' terms
    let r0 = t.clone();
    let r1 = o[0].mul(cs.namespace(|| "o0*t"), &t)?;
    let r2 = t.mul(cs.namespace(|| "t*b"), &b)?;
    let r3 = r1.mul(cs.namespace(|| "o0*t*b"), &b)?;
    let r4 = t.mul(cs.namespace(|| "t*d"), &d)?;
    let r5 = o[2].mul(cs.namespace(|| "o2*t"), &t)?;
    let r6 = o[1].mul(cs.namespace(|| "o1*t"), &t)?;
    let r7 = t.mul(cs.namespace(|| "t*s"), &s)?;
    let r8 = o[1].mul(cs.namespace(|| "o1*t*s"), &r7)?;
    let rho_prime = vec![r0, r1, r2, r3, r4, r5, r6, r7, r8];

    // sigma_i = rho_i + c * rho'_i (in Fq)
    for i in 0..9 {
      let c_rp = c_input.mul(cs.namespace(|| format!("c*rho_prime_{i}")), &rho_prime[i])?;
      let rhs = rho[i].add(cs.namespace(|| format!("rho_{i}+...")), &c_rp)?;
      cs.enforce(
        || format!("sigma_{i} check"),
        |lc| lc + rhs.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + sigma_pub[i].get_variable(),
      );
    }

    // delta_i = c * (mu_i + mu'_i) mod p, then compare in Fq
    let mu_prime = [xp.clone(), o[2].clone()];
    let c_bn = allocated_num_to_bignat(cs, "c_bn", c_input);
    for i in 0..2 {
      let mu_sum = mu[i].add(cs.namespace(|| format!("mu_sum_{i}")), &mu_prime[i])?;
      let mu_sum_bn = allocated_num_to_bignat(cs, &format!("mu_sum_bn_{i}"), mu_sum);
      let (_q, rem) = c_bn.mult_mod(cs.namespace(|| format!("delta_mod_p_{i}")), &mu_sum_bn, &modulus_p)?;
      let rem_num = bignat_to_allocated_num(cs, &format!("delta_rem_num_{i}"), &rem)?;
      cs.enforce(
        || format!("delta_{i} check"),
        |lc| lc + rem_num.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + delta_pub[i].get_variable(),
      );
    }
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
  let p = modulus_p();
  let q = modulus_q();
  let m = 16 as usize;

  let b = fq_to_nat(Fq::random(&mut rng));
  let d = fq_to_nat(Fq::random(&mut rng));
  let t = fq_to_nat(Fq::random(&mut rng));
  let xp = fq_to_nat(Fq::random(&mut rng));
  let mu = [fq_to_nat(Fq::random(&mut rng)), fq_to_nat(Fq::random(&mut rng))];
  let o = [
    fq_to_nat(Fq::random(&mut rng)),
    fq_to_nat(Fq::random(&mut rng)),
    fq_to_nat(Fq::random(&mut rng)),
  ];
  let rho: [BigInt; 9] = core::array::from_fn(|_| fq_to_nat(Fq::random(&mut rng)));
  let gamma = fq_to_nat(Fq::random(&mut rng));
  let c_fq = Fq::random(&mut rng);
  let c = fq_to_nat(c_fq);

  let e = modn(&b, &BigInt::from(m as u64));
  let s = modn(&(e + &gamma * &xp), &q);
  let rho_prime = [
    t.clone(),
    modn(&(&o[0] * &t), &q),
    modn(&(&t * &b), &q),
    modn(&(&o[0] * &t * &b), &q),
    modn(&(&t * &d), &q),
    modn(&(&o[2] * &t), &q),
    modn(&(&o[1] * &t), &q),
    modn(&(&t * &s), &q),
    modn(&(&o[1] * &t * &s), &q),
  ];
  let sigma: [BigInt; 9] = core::array::from_fn(|i| modn(&(&rho[i] + &c * &rho_prime[i]), &q));
  let mu_prime = [xp.clone(), o[2].clone()];
  let delta: [BigInt; 2] = core::array::from_fn(|i| {
    let mu_sum_q = modn(&(&mu[i] + &mu_prime[i]), &q);
    modn(&modn(&(&c * mu_sum_q), &p), &q)
  });

  let circuit = PolyTreeR0Circuit::<Fq> {
    m,
    gamma,
    sigma,
    delta,
    b,
    d,
    t,
    xp,
    mu,
    o,
    rho,
    _p: PhantomData,
  };
  let c_scalar_vec = vec![fq_to_scalar::<<E as Engine>::Scalar>(c_fq).unwrap()];

  let shape = <ShapeCS<E> as SpartanShape<E>>::r1cs_shape(&circuit).expect("shape");
  info!("PolyTreeR0: num_constraints_unpadded = {}", shape.sizes()[0]);

  let (pk, vk) = SpartanZkSNARK::<E>::setup(circuit.clone()).expect("setup failed");
  let mut prep: SpartanPrepZkSNARK<E> =
    SpartanZkSNARK::<E>::prep_prove(&pk, circuit.clone(), false).expect("prep_prove failed");
  prep.set_external_challenges(c_scalar_vec).expect("set_external_challenges failed");
  let proof = SpartanZkSNARK::<E>::prove(&pk, circuit.clone(), &prep, false).expect("prove failed");
  proof.verify(&vk).expect("verify failed");
  info!("PolyTreeR0 example completed successfully");
}