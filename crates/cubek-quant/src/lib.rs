#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[cfg(feature = "kernels")]
pub mod dequantize;

#[cfg(feature = "kernels")]
pub mod quantize;

#[cfg(feature = "kernels")]
pub mod layout;

pub use cubecl_common::quant::scheme;

/// Lloyd-Max centroid tables for codebook ([`scheme::QuantMode::Codebook`])
/// quantization. Stored values index into these; dequant is `centroid[index] * scale`.
#[cfg(feature = "kernels")]
pub(crate) mod codebook {
    /// 4-bit (`Q4F`) codebook (TQ4): 16 reconstruction levels for a unit-variance Gaussian.
    pub(crate) const Q4F: [f32; 16] = [
        -2.732590, -2.069017, -1.618046, -1.256231, -0.942340, -0.656759, -0.388048, -0.128395,
        0.128395, 0.388048, 0.656759, 0.942340, 1.256231, 1.618046, 2.069017, 2.732590,
    ];

    /// Midpoint boundary between centroid `i` and `i+1`. Centroids are sorted
    /// ascending, so `count(x >= boundary)` is the nearest-centroid index.
    pub(crate) const fn q4f_boundary(i: usize) -> f32 {
        (Q4F[i] + Q4F[i + 1]) * 0.5
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
            QuantStore::PackedU32(_) => ElemType::UInt(UIntKind::U32),
            store => panic!("Unsupported packed storage {store:?}"),
        }
    }
}
