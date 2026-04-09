//! This module implements various gadgets necessary for Nova and applications built with Nova.
mod helpers;
pub mod ecc;
pub mod nonnative;
pub mod utils;
pub mod poseidon;
pub mod less_than;

pub(crate) use helpers::OptionExt;
pub mod util_cs;
