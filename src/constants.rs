// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: MIT
// This file is part of the Spartan2 project.
// See the LICENSE file in the project root for full license information.
// Source repository: https://github.com/Microsoft/Spartan2

//! Constants for non-native field arithmetic gadgets
//! These constants are used for big number operations in circuits

/// The width of each limb in bits for big number operations
pub const BN_LIMB_WIDTH: usize = 64;

/// The number of limbs used for big number operations
pub const BN_N_LIMBS: usize = 4;

