//! Helper traits for working with bellpepper types
use bellpepper_core::SynthesisError;

/// A helper trait to provide a `.get()` method for Option types
/// This is used to maintain compatibility with Nova-style code
pub trait OptionExt<T> {
  fn get(&self) -> Result<&T, SynthesisError>;
}

impl<T> OptionExt<T> for Option<T> {
  fn get(&self) -> Result<&T, SynthesisError> {
    self.as_ref().ok_or(SynthesisError::AssignmentMissing)
  }
}

