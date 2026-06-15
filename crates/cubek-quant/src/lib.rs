#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[cfg(feature = "kernels")]
pub mod dequantize;

#[cfg(feature = "kernels")]
pub mod dequantize_tiled;

#[cfg(feature = "kernels")]
pub mod quantize;

#[cfg(feature = "kernels")]
pub mod layout;

#[cfg(feature = "kernels")]
pub mod qa_matmul;

pub use cubecl_common::quant::scheme;

/// Codebook ([`scheme::QuantMode::Codebook`]) quantization tables and helpers.
///
/// Two sub-cases, both "stored value is an index, dequant is `f(index) * scale`":
///   - **table** (TQ3/4/6 — `Q4F` here): Lloyd-Max centroids, `f(i) = centroid[i]`.
///   - **linear** (TQ8 — `Q8F`): plain affine in offset-binary, `f(i) = i - bias`.
///     bee's TQ8 stores `(code - 128) * scale`, so it's not a real codebook; the
///     linear case lets the same mode consume it directly without a 256-entry LUT.
///
/// These arrays are the *default* reconstruction tables. They are NOT baked into
/// the kernels: every codebook kernel reads its LUT (and the RHT sign pattern)
/// from a runtime buffer, so a caller can override the table entirely (see
/// [`qa_matmul::upload_codebook`] / [`qa_matmul::upload_rht_signs`], and the
/// `codebook: Handle` argument threaded through every `qa_matmul` launch). The
/// tables below are exposed via [`table`] purely as the convenient default
/// source a caller uploads when it doesn't supply its own.
#[cfg(feature = "kernels")]
pub mod codebook {
    use crate::scheme::QuantValue;

    /// 4-bit (`Q4F`) codebook (TQ4): 16 reconstruction levels for a unit-variance Gaussian.
    pub(crate) const Q4F: [f32; 16] = [
        -2.732590, -2.069017, -1.618046, -1.256231, -0.942340, -0.656759, -0.388048, -0.128395,
        0.128395, 0.388048, 0.656759, 0.942340, 1.256231, 1.618046, 2.069017, 2.732590,
    ];

    /// 6-bit (`Q6F`) codebook (TQ6): 64 Lloyd-Max levels for a unit-variance Gaussian.
    pub(crate) const Q6F: [f32; 64] = [
        -3.73971331, -3.23553866, -2.91215583, -2.66675206, -2.46556925, -2.29307792, -2.14077946,
        -2.00348979, -1.87780041, -1.76134301, -1.65240050, -1.54968499, -1.45220328, -1.35917132,
        -1.26995767, -1.18404491, -1.10100239, -1.02046671, -0.94212725, -0.86571539, -0.79099622,
        -0.71776211, -0.64582771, -0.57502585, -0.50520434, -0.43622321, -0.36795256, -0.30027058,
        -0.23306199, -0.16621658, -0.09962796, -0.03319237, 0.03319237, 0.09962796, 0.16621658,
        0.23306199, 0.30027058, 0.36795256, 0.43622321, 0.50520434, 0.57502585, 0.64582771,
        0.71776211, 0.79099622, 0.86571539, 0.94212725, 1.02046671, 1.10100239, 1.18404491,
        1.26995767, 1.35917132, 1.45220328, 1.54968499, 1.65240050, 1.76134301, 1.87780041,
        2.00348979, 2.14077946, 2.29307792, 2.46556925, 2.66675206, 2.91215583, 3.23553866,
        3.73971331,
    ];

    /// Is this value a linear (affine offset-binary) codebook rather than a table?
    pub(crate) const fn is_linear(quant: QuantValue) -> bool {
        matches!(quant, QuantValue::Q8F)
    }

    /// Number of reconstruction levels (`1 << bits`) for a table codebook.
    pub fn num_levels(quant: QuantValue) -> usize {
        1usize << quant.size_bits()
    }

    /// Default reconstruction table for `quant` (a `1 << bits`-entry slice). This
    /// is the *default* a caller uploads into the runtime codebook buffer; the
    /// kernels never reference it directly. Only table (non-linear) codebooks
    /// have one.
    pub fn table(quant: QuantValue) -> &'static [f32] {
        match quant {
            QuantValue::Q4F => &Q4F,
            QuantValue::Q6F => &Q6F,
            _ => panic!("no centroid table for {quant:?} (linear or unimplemented codebook)"),
        }
    }

    /// 1/sqrt(32) — randomized-Hadamard-transform normalization.
    pub(crate) const INV_SQRT32: f32 = 0.176_776_69;

    /// ±1 sign pattern for the 32-value RHT (the "prerot" rotation). Shared
    /// across TQ formats. Exposed as the default a caller uploads into the
    /// runtime `rht_signs` buffer; the kernels read it from that buffer.
    pub const RHT_SIGNS: [f32; 32] = [
        1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0,
        -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0,
    ];

    /// Offset-binary bias for linear codebooks: dequant is `(raw - bias)`.
    pub(crate) const fn bias(quant: QuantValue) -> i32 {
        match quant {
            QuantValue::Q8F => 128,
            _ => 0,
        }
    }

    /// Centroid `i` for a table codebook.
    pub(crate) const fn centroid(quant: QuantValue, i: usize) -> f32 {
        match quant {
            QuantValue::Q4F => Q4F[i],
            QuantValue::Q6F => Q6F[i],
            _ => panic!("no centroid table for this quant value"),
        }
    }

    /// Midpoint boundary between centroid `i` and `i+1`. Centroids are sorted
    /// ascending, so `count(x >= boundary)` is the nearest-centroid index.
    pub(crate) const fn boundary(quant: QuantValue, i: usize) -> f32 {
        (centroid(quant, i) + centroid(quant, i + 1)) * 0.5
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
