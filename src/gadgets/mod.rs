//! This module implements various gadgets necessary for Nova and applications built with Nova.
mod helpers;
pub mod ecc;
pub mod nonnative;
pub mod utils;

pub(crate) use helpers::OptionExt;
