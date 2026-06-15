#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[cfg(feature = "kernels")]
pub mod dequantize;

#[cfg(feature = "kernels")]
pub mod quantize;

#[cfg(feature = "kernels")]
pub mod layout;

#[cfg(feature = "kernels")]
pub mod qa_matmul;

pub use cubecl_common::quant::scheme;

/// Codebook ([`scheme::QuantMode::Codebook`]) quantization helpers.
///
/// Two sub-cases, both "stored value is an index, dequant is `f(index) * scale`":
///   - **table** (TQ3/4/6): Lloyd-Max centroids, `f(i) = centroid[i]`.
///   - **linear** (TQ8 — `Q8F`): plain affine in offset-binary, `f(i) = i - bias`.
///     bee's TQ8 stores `(code - 128) * scale`, so it's not a real codebook; the
///     linear case lets the same mode consume it directly without a 256-entry LUT.
///
/// The reconstruction tables and the RHT sign pattern are NOT defined here: they
/// are **comptime constants injected by the caller** (e.g. bee) via
/// [`qa_matmul::Codebook`] / [`qa_matmul::RhtSigns`], threaded through every
/// codebook launch. This module only holds the *structural* helpers (level
/// count, linear/table discrimination, offset-binary bias, RHT normalization)
/// that are not bee-specific data.
#[cfg(feature = "kernels")]
pub mod codebook {
    use crate::scheme::QuantValue;

    /// Is this value a linear (affine offset-binary) codebook rather than a table?
    pub(crate) const fn is_linear(quant: QuantValue) -> bool {
        matches!(quant, QuantValue::Q8F)
    }

    /// Number of reconstruction levels (`1 << bits`) for a table codebook.
    pub fn num_levels(quant: QuantValue) -> usize {
        1usize << quant.size_bits()
    }

    /// 1/sqrt(32) — randomized-Hadamard-transform normalization.
    pub(crate) const INV_SQRT32: f32 = 0.176_776_69;

    /// Offset-binary bias for linear codebooks: dequant is `(raw - bias)`.
    pub(crate) const fn bias(quant: QuantValue) -> i32 {
        match quant {
            QuantValue::Q8F => 128,
            _ => 0,
        }
    }
}

#[cfg(feature = "kernels")]
pub(crate) mod utils {
    use crate::scheme::{QuantLevel, QuantScheme, QuantStore};
    use cubecl::ir::{ElemType, UIntKind};

    pub(crate) fn check_block_size_compat(scheme: &QuantScheme, div: usize) {
        // Validate block size compatibility
        if let QuantScheme {
            level: QuantLevel::Block(block_size),
            ..
        } = scheme
        {
            let block_size = *block_size.as_slice().last().unwrap() as usize;
            assert!(
                block_size.is_multiple_of(div),
                "Block size must be divisible by {div}, got block_size={block_size}"
            );
        }
    }

    pub(crate) fn packed_storage_elem(scheme: &QuantScheme) -> ElemType {
        match scheme.store {
            QuantStore::PackedU32(_) | QuantStore::PackedU32Dense(_) => {
                ElemType::UInt(UIntKind::U32)
            }
            store => panic!("Unsupported packed storage {store:?}"),
        }
    }
}
