//! Less-than gadgets specialized for constant bounds.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, LinearCombination, SynthesisError};
use ff::PrimeField;
use num_bigint::BigUint;

fn alloc_boolean_num<F: PrimeField, CS: ConstraintSystem<F>>(
  mut cs: CS,
  name: impl FnOnce() -> String,
  value: Option<bool>,
) -> Result<AllocatedNum<F>, SynthesisError> {
  let b = AllocatedNum::alloc(cs.namespace(|| name()), || {
    Ok(if value.ok_or(SynthesisError::AssignmentMissing)? {
      F::ONE
    } else {
      F::ZERO
    })
  })?;

  // b * (1 - b) = 0
  cs.enforce(
    || "boolean constraint",
    |lc| lc + b.get_variable(),
    |lc| lc + CS::one() - b.get_variable(),
    |lc| lc,
  );
  Ok(b)
}

fn bits_le_from_biguint(x: &BigUint, k: usize) -> Vec<bool> {
  (0..k).map(|i| x.bit(i as u64)).collect()
}

/// Enforce that an allocated field element `x` (interpreted as an unsigned integer via its
/// bit-decomposition) satisfies `x < q`, where `q` is a public constant.
///
/// This is the optimized `eq/lt` recurrence specialized to a *constant* `q`, iterating MSB→LSB.
///
/// Returns a boolean `lt` (as an `AllocatedNum`) that is 1 iff `x < q`.
pub fn lt_constant<F: PrimeField, CS: ConstraintSystem<F>>(
  mut cs: CS,
  x: &AllocatedNum<F>,
  q: &BigUint,
) -> Result<AllocatedNum<F>, SynthesisError> {
  let k = q.bits() as usize;
  let q_bits = bits_le_from_biguint(q, k);

  // Witness bits for x, LSB->MSB.
  // Note: this constrains x < 2^k; for typical use we ensure q fits in k bits so q < 2^k.
  let x_bits_value = x.get_value().map(|fe| {
    let bytes = fe.to_repr();
    let x_nat = BigUint::from_bytes_le(bytes.as_ref());
    bits_le_from_biguint(&x_nat, k)
  });

  let mut x_bits = Vec::with_capacity(k);
  for i in 0..k {
    let b = alloc_boolean_num(
      cs.namespace(|| format!("x_bit_{i}")),
      || format!("x_bit_{i}"),
      x_bits_value.as_ref().map(|vs| vs[i]),
    )?;
    x_bits.push(b);
  }

  // Pack: sum_{i} 2^i * x_i == x
  cs.enforce(
    || "pack bits into x",
    |_| {
      let mut acc = LinearCombination::zero();
      let mut coeff = F::ONE;
      for b in &x_bits {
        acc = acc + (coeff, b.get_variable());
        coeff = coeff.double();
      }
      acc
    },
    |lc| lc + CS::one(),
    |lc| lc + x.get_variable(),
  );

  // eq_k = 1, lt_k = 0
  let mut eq = AllocatedNum::alloc(cs.namespace(|| "eq_k"), || Ok(F::ONE))?;
  cs.enforce(
    || "eq_k is 1",
    |lc| lc + CS::one(),
    |lc| lc + CS::one(),
    |lc| lc + eq.get_variable(),
  );
  let mut lt = AllocatedNum::alloc(cs.namespace(|| "lt_k"), || Ok(F::ZERO))?;
  cs.enforce(
    || "lt_k is 0",
    |lc| lc,
    |lc| lc,
    |lc| lc + lt.get_variable(),
  );

  // MSB -> LSB
  for i in (0..k).rev() {
    let x_i = &x_bits[i];
    let q_i = q_bits[i];

    if !q_i {
      // eq_i = eq_{i+1} * (1 - x_i)
      let eq_next = AllocatedNum::alloc(cs.namespace(|| format!("eq_{i}")), || {
        let eqv = eq.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        let xv = x_i.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        Ok(eqv * (F::ONE - xv))
      })?;
      cs.enforce(
        || format!("eq update when q_{i}=0"),
        |lc| lc + eq.get_variable(),
        |lc| lc + CS::one() - x_i.get_variable(),
        |lc| lc + eq_next.get_variable(),
      );
      eq = eq_next;
      // lt unchanged
    } else {
      // eq_i = eq_{i+1} * x_i
      let eq_next = AllocatedNum::alloc(cs.namespace(|| format!("eq_{i}")), || {
        let eqv = eq.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        let xv = x_i.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        Ok(eqv * xv)
      })?;
      cs.enforce(
        || format!("eq update when q_{i}=1"),
        |lc| lc + eq.get_variable(),
        |lc| lc + x_i.get_variable(),
        |lc| lc + eq_next.get_variable(),
      );

      // t = eq_{i+1} * (1 - x_i)
      let t = AllocatedNum::alloc(cs.namespace(|| format!("t_{i}")), || {
        let eqv = eq.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        let xv = x_i.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        Ok(eqv * (F::ONE - xv))
      })?;
      cs.enforce(
        || format!("t update when q_{i}=1"),
        |lc| lc + eq.get_variable(),
        |lc| lc + CS::one() - x_i.get_variable(),
        |lc| lc + t.get_variable(),
      );

      // lt_i = lt_{i+1} OR t via (1-lt_prev)*(1-t) = 1-lt_next
      let lt_next = AllocatedNum::alloc(cs.namespace(|| format!("lt_{i}")), || {
        let ltv = lt.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        let tv = t.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        Ok(if ltv == F::ONE || tv == F::ONE { F::ONE } else { F::ZERO })
      })?;
      cs.enforce(
        || format!("lt or when q_{i}=1"),
        |lc| lc + CS::one() - lt.get_variable(),
        |lc| lc + CS::one() - t.get_variable(),
        |lc| lc + CS::one() - lt_next.get_variable(),
      );

      eq = eq_next;
      lt = lt_next;
    }
  }

  Ok(lt)
}

