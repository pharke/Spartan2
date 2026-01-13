//! This module implements various elliptic curve gadgets
#![allow(non_snake_case)]
use crate::{
  constants::{BN_LIMB_WIDTH as LIMB_WIDTH, BN_N_LIMBS as N_LIMBS},
  gadgets::{
    nonnative::{bignat::BigNat, util::f_to_nat},
    utils::{
      alloc_bignat_constant, alloc_num_equals, alloc_one, alloc_zero, conditionally_select,
      conditionally_select2, conditionally_select_bignat, select_num_or_one, select_num_or_zero,
      select_num_or_zero2, select_one_or_diff2, select_one_or_num2, select_zero_or_num2,
      field_switch,
    },
    OptionExt,
  },
  traits::{Engine, Group},
};
use bellpepper_core::{
  num::AllocatedNum,
  boolean::{AllocatedBit, Boolean},
  ConstraintSystem, SynthesisError,
};
use ff::{Field, PrimeField, PrimeFieldBits};
use num_bigint::BigInt;
use std::marker::PhantomData;

/// `AllocatedPoint` provides an elliptic curve abstraction inside a circuit.
/// It works in the scalar field `E::Scalar` of the proof engine `E`,
/// but uses curve parameters from a separate group `G` where `G::Base` can be converted to/from `E::Scalar`.
/// This allows proving statements about curves defined over `E::Scalar` (e.g., P256)
/// while using `E::Scalar` as the constraint system field (e.g., T256::Scalar).
///
/// Note: In halo2curves, `p256::Base` and `t256::Scalar` are the same type, but Rust's type system
/// treats them as different. We use a helper trait to handle this conversion.
#[derive(Clone)]
pub struct AllocatedPoint<E: Engine, G: Group>
where
  G::Base: PrimeField + PrimeFieldBits,
  E::Scalar: PrimeField + PrimeFieldBits,
{
  /// The x-coordinate of the point
  pub x: AllocatedNum<E::Scalar>,
  /// The y-coordinate of the point
  pub y: AllocatedNum<E::Scalar>,
  /// Whether the point is at infinity (1 if infinity, 0 otherwise)
  pub is_infinity: AllocatedNum<E::Scalar>,
  _marker: PhantomData<G>,
}

impl<E, G> AllocatedPoint<E, G>
where
  E: Engine,
  G: Group,
  G::Base: PrimeField + PrimeFieldBits,
  E::Scalar: PrimeField + PrimeFieldBits,
{
  /// Allocates a new point on the curve using coordinates provided by `coords`.
  /// If coords = None, it allocates the default infinity point
  pub fn alloc<CS: ConstraintSystem<E::Scalar>>(
    mut cs: CS,
    coords: Option<(E::Scalar, E::Scalar, bool)>,
  ) -> Result<Self, SynthesisError> {
    let x = AllocatedNum::alloc(cs.namespace(|| "x"), || {
      Ok(coords.map_or(E::Scalar::ZERO, |c| c.0))
    })?;
    let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
      Ok(coords.map_or(E::Scalar::ZERO, |c| c.1))
    })?;
    let is_infinity = AllocatedNum::alloc(cs.namespace(|| "is_infinity"), || {
      Ok(if coords.map_or(true, |c| c.2) {
        E::Scalar::ONE
      } else {
        E::Scalar::ZERO
      })
    })?;
    cs.enforce(
      || "is_infinity is bit",
      |lc| lc + is_infinity.get_variable(),
      |lc| lc + CS::one() - is_infinity.get_variable(),
      |lc| lc,
    );

    Ok(AllocatedPoint {
      x,
      y,
      is_infinity,
      _marker: PhantomData,
    })
  }

  /// checks if `self` is on the curve or if it is infinity
  pub fn check_on_curve<CS>(&self, mut cs: CS) -> Result<(), SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    // check that (x,y) is on the curve if it is not infinity
    // we will check that (1- is_infinity) * y^2 = (1-is_infinity) * (x^3 + Ax + B)
    // note that is_infinity is already restricted to be in the set {0, 1}
    let y_square = self.y.square(cs.namespace(|| "y_square"))?;
    let x_square = self.x.square(cs.namespace(|| "x_square"))?;
    let x_cube = self.x.mul(cs.namespace(|| "x_cube"), &x_square)?;

    // Convert curve parameters from G::Base to E::Scalar
    let curve_a = field_switch::<G::Base, E::Scalar>(G::group_params().0);
    let curve_b = field_switch::<G::Base, E::Scalar>(G::group_params().1);

    let rhs = AllocatedNum::alloc(cs.namespace(|| "rhs"), || {
      let is_inf = self.is_infinity.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      if is_inf == E::Scalar::ONE {
        Ok(E::Scalar::ZERO)
      } else {
        let x_cube_val = x_cube.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        let x_val = self.x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        Ok(
          x_cube_val
            + x_val * curve_a
            + curve_b,
        )
      }
    })?;

    cs.enforce(
      || "rhs = (1-is_infinity) * (x^3 + Ax + B)",
      |lc| {
        lc + x_cube.get_variable()
          + (curve_a, self.x.get_variable())
          + (curve_b, CS::one())
      },
      |lc| lc + CS::one() - self.is_infinity.get_variable(),
      |lc| lc + rhs.get_variable(),
    );

    // check that (1-infinity) * y_square = rhs
    cs.enforce(
      || "check that y_square * (1 - is_infinity) = rhs",
      |lc| lc + y_square.get_variable(),
      |lc| lc + CS::one() - self.is_infinity.get_variable(),
      |lc| lc + rhs.get_variable(),
    );

    Ok(())
  }

  /// Allocates a default point on the curve, set to the identity point.
  pub fn default<CS: ConstraintSystem<E::Scalar>>(mut cs: CS) -> Result<Self, SynthesisError> {
    let zero = alloc_zero(cs.namespace(|| "zero"));
    let one = alloc_one(cs.namespace(|| "one"));

    Ok(AllocatedPoint {
      x: zero.clone(),
      y: zero,
      is_infinity: one,
      _marker: PhantomData,
    })
  }

  /// Negates the provided point
  pub fn negate<CS: ConstraintSystem<E::Scalar>>(&self, mut cs: CS) -> Result<Self, SynthesisError> {
    let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
      Ok(-self.y.get_value().ok_or(SynthesisError::AssignmentMissing)?)
    })?;

    cs.enforce(
      || "check y = - self.y",
      |lc| lc + self.y.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc - y.get_variable(),
    );

    Ok(Self {
      x: self.x.clone(),
      y,
      is_infinity: self.is_infinity.clone(),
      _marker: PhantomData,
    })
  }

  /// Add two points (may be equal)
  pub fn add<CS: ConstraintSystem<E::Scalar>>(
    &self,
    mut cs: CS,
    other: &AllocatedPoint<E, G>,
  ) -> Result<Self, SynthesisError> {
    // Compute boolean equal indicating if self = other

    let equal_x = alloc_num_equals(
      cs.namespace(|| "check self.x == other.x"),
      &self.x,
      &other.x,
    )?;

    let equal_y = alloc_num_equals(
      cs.namespace(|| "check self.y == other.y"),
      &self.y,
      &other.y,
    )?;

    // Compute the result of the addition and the result of double self
    let result_from_add = self.add_internal(cs.namespace(|| "add internal"), other, &equal_x)?;
    let result_from_double = self.double(cs.namespace(|| "double"))?;

    // Output:
    // If (self == other) {
    //  return double(self)
    // }else {
    //  if (self.x == other.x){
    //      return infinity [negation]
    //  } else {
    //      return add(self, other)
    //  }
    // }
    let result_for_equal_x = AllocatedPoint::select_point_or_infinity(
      cs.namespace(|| "equal_y ? result_from_double : infinity"),
      &result_from_double,
      &Boolean::from(equal_y),
    )?;

    AllocatedPoint::conditionally_select(
      cs.namespace(|| "equal ? result_from_double : result_from_add"),
      &result_for_equal_x,
      &result_from_add,
      &Boolean::from(equal_x),
    )
  }

  /// Adds other point to this point and returns the result. Assumes that the two points are
  /// different and that both `other.is_infinity` and `this.is_infinity` are bits
  pub fn add_internal<CS: ConstraintSystem<E::Scalar>>(
    &self,
    mut cs: CS,
    other: &AllocatedPoint<E, G>,
    equal_x: &AllocatedBit,
  ) -> Result<Self, SynthesisError> {
    //************************************************************************/
    // lambda = (other.y - self.y) * (other.x - self.x).invert().unwrap();
    //************************************************************************/
    // First compute (other.x - self.x).inverse()
    // If either self or other are the infinity point or self.x = other.x  then compute bogus values
    // Specifically,
    // x_diff = self != inf && other != inf && self.x == other.x ? (other.x - self.x) : 1

    // Compute self.is_infinity OR other.is_infinity =
    // NOT(NOT(self.is_ifninity) AND NOT(other.is_infinity))
    let at_least_one_inf = AllocatedNum::alloc(cs.namespace(|| "at least one inf"), || {
      Ok(
        E::Scalar::ONE
          - (E::Scalar::ONE - *self.is_infinity.get_value().get()?)
            * (E::Scalar::ONE - *other.is_infinity.get_value().get()?),
      )
    })?;
    cs.enforce(
      || "1 - at least one inf = (1-self.is_infinity) * (1-other.is_infinity)",
      |lc| lc + CS::one() - self.is_infinity.get_variable(),
      |lc| lc + CS::one() - other.is_infinity.get_variable(),
      |lc| lc + CS::one() - at_least_one_inf.get_variable(),
    );

    // Now compute x_diff_is_actual = at_least_one_inf OR equal_x
    let x_diff_is_actual =
      AllocatedNum::alloc(cs.namespace(|| "allocate x_diff_is_actual"), || {
        let equal_x_val = equal_x.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        let at_least_one_inf_val = at_least_one_inf.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        Ok(if equal_x_val {
          E::Scalar::ONE
        } else {
          at_least_one_inf_val
        })
      })?;
    cs.enforce(
      || "1 - x_diff_is_actual = (1-equal_x) * (1-at_least_one_inf)",
      |lc| lc + CS::one() - at_least_one_inf.get_variable(),
      |lc| lc + CS::one() - equal_x.get_variable(),
      |lc| lc + CS::one() - x_diff_is_actual.get_variable(),
    );

    // x_diff = 1 if either self.is_infinity or other.is_infinity or self.x = other.x else self.x -
    // other.x
    let x_diff = select_one_or_diff2(
      cs.namespace(|| "Compute x_diff"),
      &other.x,
      &self.x,
      &x_diff_is_actual,
    )?;

    let lambda = AllocatedNum::alloc(cs.namespace(|| "lambda"), || {
      let x_diff_inv = if *x_diff_is_actual.get_value().get()? == E::Scalar::ONE {
        // Set to default
        E::Scalar::ONE
      } else {
        // Set to the actual inverse
        (*other.x.get_value().get()? - *self.x.get_value().get()?)
          .invert()
          .unwrap()
      };

      Ok((*other.y.get_value().get()? - *self.y.get_value().get()?) * x_diff_inv)
    })?;
    cs.enforce(
      || "Check that lambda is correct",
      |lc| lc + lambda.get_variable(),
      |lc| lc + x_diff.get_variable(),
      |lc| lc + other.y.get_variable() - self.y.get_variable(),
    );

    //************************************************************************/
    // x = lambda * lambda - self.x - other.x;
    //************************************************************************/
    let x = AllocatedNum::alloc(cs.namespace(|| "x"), || {
      Ok(
        *lambda.get_value().get()? * lambda.get_value().get()?
          - *self.x.get_value().get()?
          - *other.x.get_value().get()?,
      )
    })?;
    cs.enforce(
      || "check that x is correct",
      |lc| lc + lambda.get_variable(),
      |lc| lc + lambda.get_variable(),
      |lc| lc + x.get_variable() + self.x.get_variable() + other.x.get_variable(),
    );

    //************************************************************************/
    // y = lambda * (self.x - x) - self.y;
    //************************************************************************/
    let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
      Ok(
        *lambda.get_value().get()? * (*self.x.get_value().get()? - *x.get_value().get()?)
          - *self.y.get_value().get()?,
      )
    })?;

    cs.enforce(
      || "Check that y is correct",
      |lc| lc + lambda.get_variable(),
      |lc| lc + self.x.get_variable() - x.get_variable(),
      |lc| lc + y.get_variable() + self.y.get_variable(),
    );

    //************************************************************************/
    // We only return the computed x, y if neither of the points is infinity and self.x != other.y
    // if self.is_infinity return other.clone()
    // elif other.is_infinity return self.clone()
    // elif self.x == other.x return infinity
    // Otherwise return the computed points.
    //************************************************************************/
    // Now compute the output x

    let x1 = conditionally_select2(
      cs.namespace(|| "x1 = other.is_infinity ? self.x : x"),
      &self.x,
      &x,
      &other.is_infinity,
    )?;

    let x = conditionally_select2(
      cs.namespace(|| "x = self.is_infinity ? other.x : x1"),
      &other.x,
      &x1,
      &self.is_infinity,
    )?;

    let y1 = conditionally_select2(
      cs.namespace(|| "y1 = other.is_infinity ? self.y : y"),
      &self.y,
      &y,
      &other.is_infinity,
    )?;

    let y = conditionally_select2(
      cs.namespace(|| "y = self.is_infinity ? other.y : y1"),
      &other.y,
      &y1,
      &self.is_infinity,
    )?;

    let is_infinity1 = select_num_or_zero2(
      cs.namespace(|| "is_infinity1 = other.is_infinity ? self.is_infinity : 0"),
      &self.is_infinity,
      &other.is_infinity,
    )?;

    let is_infinity = conditionally_select2(
      cs.namespace(|| "is_infinity = self.is_infinity ? other.is_infinity : is_infinity1"),
      &other.is_infinity,
      &is_infinity1,
      &self.is_infinity,
    )?;

    Ok(Self {
      x,
      y,
      is_infinity,
      _marker: PhantomData,
    })
  }

  /// Doubles the supplied point.
  pub fn double<CS: ConstraintSystem<E::Scalar>>(&self, mut cs: CS) -> Result<Self, SynthesisError> {
    //*************************************************************/
    // lambda = (E::Scalar::from(3) * self.x * self.x + G::A())
    //  * (E::Scalar::from(2)) * self.y).invert().unwrap();
    /*************************************************************/

    // Compute tmp = (E::Scalar::ONE + E::Scalar::ONE)* self.y ? self != inf : 1
    let tmp_actual = AllocatedNum::alloc(cs.namespace(|| "tmp_actual"), || {
      Ok(*self.y.get_value().get()? + *self.y.get_value().get()?)
    })?;
    cs.enforce(
      || "check tmp_actual",
      |lc| lc + CS::one() + CS::one(),
      |lc| lc + self.y.get_variable(),
      |lc| lc + tmp_actual.get_variable(),
    );

    let tmp = select_one_or_num2(cs.namespace(|| "tmp"), &tmp_actual, &self.is_infinity)?;

    // Now compute lambda as (E::Scalar::from(3) * self.x * self.x + G::A()) * tmp_inv

    let prod_1 = AllocatedNum::alloc(cs.namespace(|| "alloc prod 1"), || {
      Ok(E::Scalar::from(3) * self.x.get_value().get()? * self.x.get_value().get()?)
    })?;
    cs.enforce(
      || "Check prod 1",
      |lc| lc + (E::Scalar::from(3), self.x.get_variable()),
      |lc| lc + self.x.get_variable(),
      |lc| lc + prod_1.get_variable(),
    );

    // Convert curve parameter from G::Base to E::Scalar
    let curve_a = field_switch::<G::Base, E::Scalar>(G::group_params().0);

    let lambda = AllocatedNum::alloc(cs.namespace(|| "alloc lambda"), || {
      let tmp_inv = if *self.is_infinity.get_value().get()? == E::Scalar::ONE {
        // Return default value 1
        E::Scalar::ONE
      } else {
        // Return the actual inverse
        (*tmp.get_value().get()?).invert().unwrap()
      };

      Ok(tmp_inv * (*prod_1.get_value().get()? + curve_a))
    })?;

    cs.enforce(
      || "Check lambda",
      |lc| lc + tmp.get_variable(),
      |lc| lc + lambda.get_variable(),
      |lc| lc + prod_1.get_variable() + (curve_a, CS::one()),
    );

    /*************************************************************/
    //          x = lambda * lambda - self.x - self.x;
    /*************************************************************/

    let x = AllocatedNum::alloc(cs.namespace(|| "x"), || {
      Ok(
        ((*lambda.get_value().get()?) * (*lambda.get_value().get()?))
          - *self.x.get_value().get()?
          - self.x.get_value().get()?,
      )
    })?;
    cs.enforce(
      || "Check x",
      |lc| lc + lambda.get_variable(),
      |lc| lc + lambda.get_variable(),
      |lc| lc + x.get_variable() + self.x.get_variable() + self.x.get_variable(),
    );

    /*************************************************************/
    //        y = lambda * (self.x - x) - self.y;
    /*************************************************************/

    let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
      Ok(
        (*lambda.get_value().get()?) * (*self.x.get_value().get()? - x.get_value().get()?)
          - self.y.get_value().get()?,
      )
    })?;
    cs.enforce(
      || "Check y",
      |lc| lc + lambda.get_variable(),
      |lc| lc + self.x.get_variable() - x.get_variable(),
      |lc| lc + y.get_variable() + self.y.get_variable(),
    );

    /*************************************************************/
    // Only return the computed x and y if the point is not infinity
    /*************************************************************/

    // x
    let x = select_zero_or_num2(cs.namespace(|| "final x"), &x, &self.is_infinity)?;

    // y
    let y = select_zero_or_num2(cs.namespace(|| "final y"), &y, &self.is_infinity)?;

    // is_infinity
    let is_infinity = self.is_infinity.clone();

    Ok(Self {
      x,
      y,
      is_infinity,
      _marker: PhantomData,
    })
  }

  /// A gadget for scalar multiplication, optimized to use incomplete addition law.
  /// The optimization here is analogous to <https://github.com/arkworks-rs/r1cs-std/blob/6d64f379a27011b3629cf4c9cb38b7b7b695d5a0/src/groups/curves/short_weierstrass/mod.rs#L295>,
  /// except we use complete addition law over affine coordinates instead of projective coordinates for the tail bits
  pub fn scalar_mul<CS: ConstraintSystem<E::Scalar>>(
    &self,
    mut cs: CS,
    scalar_bits: &[AllocatedBit],
  ) -> Result<Self, SynthesisError> {
    let split_len = core::cmp::min(scalar_bits.len(), (E::Scalar::NUM_BITS - 2) as usize);
    let (incomplete_bits, complete_bits) = scalar_bits.split_at(split_len);

    // we convert AllocatedPoint into AllocatedPointNonInfinity; we deal with the case where self.is_infinity = 1 below
    let mut p = AllocatedPointNonInfinity::<E, G>::from_allocated_point(self);

    // we assume the first bit to be 1, so we must initialize acc to self and double it
    // we remove this assumption below
    let mut acc = p;
    p = acc.double_incomplete(cs.namespace(|| "double"))?;

    // perform the double-and-add loop to compute the scalar mul using incomplete addition law
    for (i, bit) in incomplete_bits.iter().enumerate().skip(1) {
      let temp = acc.add_incomplete(cs.namespace(|| format!("add {i}")), &p)?;
      acc = AllocatedPointNonInfinity::<E, G>::conditionally_select(
        cs.namespace(|| format!("acc_iteration_{i}")),
        &temp,
        &acc,
        &Boolean::from(bit.clone()),
      )?;

      p = p.double_incomplete(cs.namespace(|| format!("double {i}")))?;
    }

    // convert back to AllocatedPoint
    let res = {
      // we set acc.is_infinity = self.is_infinity
      let acc = acc.to_allocated_point(&self.is_infinity)?;

      // we remove the initial slack if bits[0] is as not as assumed (i.e., it is not 1)
      let acc_minus_initial = {
        let neg = self.negate(cs.namespace(|| "negate"))?;
        acc.add(cs.namespace(|| "res minus self"), &neg)
      }?;

      Self::conditionally_select(
        cs.namespace(|| "remove slack if necessary"),
        &acc,
        &acc_minus_initial,
        &Boolean::from(scalar_bits[0].clone()),
      )?
    };

    // when self.is_infinity = 1, return the default point, else return res
    // we already set res.is_infinity to be self.is_infinity, so we do not need to set it here
    let default = Self::default(cs.namespace(|| "default"))?;
    let x = conditionally_select2(
      cs.namespace(|| "check if self.is_infinity is zero (x)"),
      &default.x,
      &res.x,
      &self.is_infinity,
    )?;

    let y = conditionally_select2(
      cs.namespace(|| "check if self.is_infinity is zero (y)"),
      &default.y,
      &res.y,
      &self.is_infinity,
    )?;

    // we now perform the remaining scalar mul using complete addition law
    let mut acc = Self {
      x,
      y,
      is_infinity: res.is_infinity,
      _marker: PhantomData,
    };
    let mut p_complete = p.to_allocated_point(&self.is_infinity)?;

    for (i, bit) in complete_bits.iter().enumerate() {
      let temp = acc.add(cs.namespace(|| format!("add_complete {i}")), &p_complete)?;
      acc = Self::conditionally_select(
        cs.namespace(|| format!("acc_complete_iteration_{i}")),
        &temp,
        &acc,
        &Boolean::from(bit.clone()),
      )?;

      p_complete = p_complete.double(cs.namespace(|| format!("double_complete {i}")))?;
    }

    Ok(acc)
  }

  /// If condition outputs a otherwise outputs b
  pub fn conditionally_select<CS: ConstraintSystem<E::Scalar>>(
    mut cs: CS,
    a: &Self,
    b: &Self,
    condition: &Boolean,
  ) -> Result<Self, SynthesisError> {
    let x = conditionally_select(cs.namespace(|| "select x"), &a.x, &b.x, condition)?;

    let y = conditionally_select(cs.namespace(|| "select y"), &a.y, &b.y, condition)?;

    let is_infinity = conditionally_select(
      cs.namespace(|| "select is_infinity"),
      &a.is_infinity,
      &b.is_infinity,
      condition,
    )?;

    Ok(Self {
      x,
      y,
      is_infinity,
      _marker: PhantomData,
    })
  }

  /// If condition outputs a otherwise infinity
  pub fn select_point_or_infinity<CS: ConstraintSystem<E::Scalar>>(
    mut cs: CS,
    a: &Self,
    condition: &Boolean,
  ) -> Result<Self, SynthesisError> {
    let x = select_num_or_zero(cs.namespace(|| "select x"), &a.x, condition)?;

    let y = select_num_or_zero(cs.namespace(|| "select y"), &a.y, condition)?;

    let is_infinity = select_num_or_one(
      cs.namespace(|| "select is_infinity"),
      &a.is_infinity,
      condition,
    )?;

    Ok(Self {
      x,
      y,
      is_infinity,
      _marker: PhantomData,
    })
  }
}

#[derive(Clone)]
/// `AllocatedPoint` but one that is guaranteed to be not infinity
pub struct AllocatedPointNonInfinity<E: Engine, G: Group>
where
  G::Base: PrimeField + PrimeFieldBits,
  E::Scalar: PrimeField + PrimeFieldBits,
{
  x: AllocatedNum<E::Scalar>,
  y: AllocatedNum<E::Scalar>,
  _marker: PhantomData<G>,
}

impl<E: Engine, G: Group> AllocatedPointNonInfinity<E, G>
where
  G::Base: PrimeField + PrimeFieldBits,
  E::Scalar: PrimeField + PrimeFieldBits,
{
  /// Turns an `AllocatedPoint` into an `AllocatedPointNonInfinity` (assumes it is not infinity)
  pub fn from_allocated_point(p: &AllocatedPoint<E, G>) -> Self {
    Self {
      x: p.x.clone(),
      y: p.y.clone(),
      _marker: PhantomData,
    }
  }

  /// Returns an `AllocatedPoint` from an `AllocatedPointNonInfinity`
  pub fn to_allocated_point(
    &self,
    is_infinity: &AllocatedNum<E::Scalar>,
  ) -> Result<AllocatedPoint<E, G>, SynthesisError> {
    Ok(AllocatedPoint {
      x: self.x.clone(),
      y: self.y.clone(),
      is_infinity: is_infinity.clone(),
      _marker: PhantomData,
    })
  }

  /// Add two points assuming self != +/- other
  pub fn add_incomplete<CS>(&self, mut cs: CS, other: &Self) -> Result<Self, SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    // allocate a free variable that an honest prover sets to lambda = (y2-y1)/(x2-x1)
    let lambda = AllocatedNum::alloc(cs.namespace(|| "lambda"), || {
      if *other.x.get_value().get()? == *self.x.get_value().get()? {
        Ok(E::Scalar::ONE)
      } else {
        Ok(
          (*other.y.get_value().get()? - *self.y.get_value().get()?)
            * (*other.x.get_value().get()? - *self.x.get_value().get()?)
              .invert()
              .unwrap(),
        )
      }
    })?;
    cs.enforce(
      || "Check that lambda is computed correctly",
      |lc| lc + lambda.get_variable(),
      |lc| lc + other.x.get_variable() - self.x.get_variable(),
      |lc| lc + other.y.get_variable() - self.y.get_variable(),
    );

    //************************************************************************/
    // x = lambda * lambda - self.x - other.x;
    //************************************************************************/
    let x = AllocatedNum::alloc(cs.namespace(|| "x"), || {
      Ok(
        *lambda.get_value().get()? * lambda.get_value().get()?
          - *self.x.get_value().get()?
          - *other.x.get_value().get()?,
      )
    })?;
    cs.enforce(
      || "check that x is correct",
      |lc| lc + lambda.get_variable(),
      |lc| lc + lambda.get_variable(),
      |lc| lc + x.get_variable() + self.x.get_variable() + other.x.get_variable(),
    );

    //************************************************************************/
    // y = lambda * (self.x - x) - self.y;
    //************************************************************************/
    let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
      Ok(
        *lambda.get_value().get()? * (*self.x.get_value().get()? - *x.get_value().get()?)
          - *self.y.get_value().get()?,
      )
    })?;

    cs.enforce(
      || "Check that y is correct",
      |lc| lc + lambda.get_variable(),
      |lc| lc + self.x.get_variable() - x.get_variable(),
      |lc| lc + y.get_variable() + self.y.get_variable(),
    );

    Ok(Self {
      x,
      y,
      _marker: PhantomData,
    })
  }

  /// doubles the point; since this is called with a point not at infinity, it is guaranteed to be not infinity
  pub fn double_incomplete<CS: ConstraintSystem<E::Scalar>>(
    &self,
    mut cs: CS,
  ) -> Result<Self, SynthesisError> {
    // lambda = (3 x^2 + a) / 2 * y

    let x_sq = self.x.square(cs.namespace(|| "x_sq"))?;

    // Convert curve parameter from G::Base to E::Scalar
    let curve_a = field_switch::<G::Base, E::Scalar>(G::group_params().0);

    let lambda = AllocatedNum::alloc(cs.namespace(|| "lambda"), || {
      let n = E::Scalar::from(3) * x_sq.get_value().get()? + curve_a;
      let d = E::Scalar::from(2) * *self.y.get_value().get()?;
      if d == E::Scalar::ZERO {
        Ok(E::Scalar::ONE)
      } else {
        Ok(n * d.invert().unwrap())
      }
    })?;
    cs.enforce(
      || "Check that lambda is computed correctly",
      |lc| lc + lambda.get_variable(),
      |lc| lc + (E::Scalar::from(2), self.y.get_variable()),
      |lc| lc + (E::Scalar::from(3), x_sq.get_variable()) + (curve_a, CS::one()),
    );

    let x = AllocatedNum::alloc(cs.namespace(|| "x"), || {
      Ok(
        *lambda.get_value().get()? * *lambda.get_value().get()?
          - *self.x.get_value().get()?
          - *self.x.get_value().get()?,
      )
    })?;

    cs.enforce(
      || "check that x is correct",
      |lc| lc + lambda.get_variable(),
      |lc| lc + lambda.get_variable(),
      |lc| lc + x.get_variable() + (E::Scalar::from(2), self.x.get_variable()),
    );

    let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
      Ok(
        *lambda.get_value().get()? * (*self.x.get_value().get()? - *x.get_value().get()?)
          - *self.y.get_value().get()?,
      )
    })?;

    cs.enforce(
      || "Check that y is correct",
      |lc| lc + lambda.get_variable(),
      |lc| lc + self.x.get_variable() - x.get_variable(),
      |lc| lc + y.get_variable() + self.y.get_variable(),
    );

    Ok(Self {
      x,
      y,
      _marker: PhantomData,
    })
  }

  /// If condition outputs a otherwise outputs b
  pub fn conditionally_select<CS: ConstraintSystem<E::Scalar>>(
    mut cs: CS,
    a: &Self,
    b: &Self,
    condition: &Boolean,
  ) -> Result<Self, SynthesisError> {
    let x = conditionally_select(cs.namespace(|| "select x"), &a.x, &b.x, condition)?;
    let y = conditionally_select(cs.namespace(|| "select y"), &a.y, &b.y, condition)?;

    Ok(Self {
      x,
      y,
      _marker: PhantomData,
    })
  }
}

// `AllocatedNonnativePoint`s are points on an elliptic curve E'. We use the scalar field
// of another curve E (specified as the group G) to prove things about points on E'.
// `AllocatedNonnativePoint`s are always represented as affine coordinates.
#[derive(Clone, Debug)]
struct AllocatedNonnativePoint<E: Engine> {
  pub(crate) x: BigNat<E::Scalar>,
  pub(crate) y: BigNat<E::Scalar>,
  pub(crate) is_infinity: AllocatedNum<E::Scalar>,
}

#[allow(dead_code)]
impl<E: Engine> AllocatedNonnativePoint<E> {
  pub fn alloc<CS: ConstraintSystem<E::Scalar>>(
    mut cs: CS,
    coords: Option<(E::Base, E::Base, bool)>,
  ) -> Result<Self, SynthesisError> {
    let x = BigNat::alloc_from_nat(
      cs.namespace(|| "x as BigNat"),
      || Ok(coords.map_or(f_to_nat(&E::Base::ZERO), |v| f_to_nat(&v.0))),
      LIMB_WIDTH,
      N_LIMBS,
    )?;

    let y = BigNat::alloc_from_nat(
      cs.namespace(|| "y as BigNat"),
      || Ok(coords.map_or(f_to_nat(&E::Base::ZERO), |v| f_to_nat(&v.1))),
      LIMB_WIDTH,
      N_LIMBS,
    )?;

    let is_infinity = AllocatedNum::alloc(cs.namespace(|| "is_infinity"), || {
      Ok(if coords.map_or(true, |c| c.2) {
        E::Scalar::ONE
      } else {
        E::Scalar::ZERO
      })
    })?;

    cs.enforce(
      || "is_infinity is bit",
      |lc| lc + is_infinity.get_variable(),
      |lc| lc + CS::one() - is_infinity.get_variable(),
      |lc| lc,
    );

    Ok(Self { x, y, is_infinity })
  }

  /// Allocates a default point on the curve, set to the point at infinity.
  pub fn default<CS>(mut cs: CS) -> Result<Self, SynthesisError>
  where
    CS: ConstraintSystem<E::Scalar>,
  {
    let one = alloc_one(cs.namespace(|| "one"));
    let zero = alloc_bignat_constant(
      cs.namespace(|| "zero"),
      &BigInt::from(0),
      LIMB_WIDTH,
      N_LIMBS,
    )?;

    Ok(AllocatedNonnativePoint {
      x: zero.clone(),
      y: zero,
      is_infinity: one,
    })
  }

  // NOTE: This function requires RO2Circuit which is not available in spartan2
  // If needed, this can be reimplemented using spartan2's transcript system
  // /// Absorb the provided instance in the RO
  // pub fn absorb_in_ro<CS: ConstraintSystem<E::Scalar>>(
  //   &self,
  //   mut cs: CS,
  //   ro: &mut E::RO2Circuit,
  // ) -> Result<(), SynthesisError> {
  //   for (i, limb) in self.x.as_limbs().iter().enumerate() {
  //     let limb_num =
  //       limb.as_allocated_num(cs.namespace(|| format!("convert limb {i} of num x")))?;
  //     ro.absorb(&limb_num);
  //   }
  //
  //   for (i, limb) in self.y.as_limbs().iter().enumerate() {
  //     let limb_num =
  //       limb.as_allocated_num(cs.namespace(|| format!("convert limb {i} of num y")))?;
  //     ro.absorb(&limb_num);
  //   }
  //
  //   ro.absorb(&self.is_infinity);
  //
  //   Ok(())
  // }

  /// If condition outputs a otherwise outputs b
  pub fn conditionally_select<CS: ConstraintSystem<E::Scalar>>(
    mut cs: CS,
    a: &Self,
    b: &Self,
    condition: &Boolean,
  ) -> Result<Self, SynthesisError> {
    let x = conditionally_select_bignat(cs.namespace(|| "select x"), &a.x, &b.x, condition)?;
    let y = conditionally_select_bignat(cs.namespace(|| "select y"), &a.y, &b.y, condition)?;
    let is_infinity = conditionally_select(
      cs.namespace(|| "select is_infinity"),
      &a.is_infinity,
      &b.is_infinity,
      condition,
    )?;

    Ok(Self { x, y, is_infinity })
  }
}
