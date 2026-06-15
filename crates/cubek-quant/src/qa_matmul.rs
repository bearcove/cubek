//! QA (quant-activation) matmul for TQ6: both operands are TQ6 codebook-quantized
//! in randomized-Hadamard (prerot) space. This is bee's `tq_projection_qa` path —
//! the production decode forward.
//!
//! For each output `C[m,n] = Σ_k dequant(A)[m,k] · dequant(W)[n,k]`. Because both
//! operands share the centroid table, the inner product over a 16-code unit is
//! `(Σ_j centroid[aᵢⱼ]·centroid[wᵢⱼ]) · scale_a · scale_b` — i.e. the LUT-outer-product
//! structure (the centroid·centroid product is what bee's `tq6x6_lut` precomputes),
//! with the per-half-block scales factored out of the sum. The rotation cancels
//! because both operands live in the same prerot space.
//!
//! Codes are dense TQ6 (`PackedU32Dense`): code `j` at bit `6j`, 16 codes per
//! `lcm(6,32)=96`-bit unit = 3 u32. One thread per output element (this is the
//! correctness reference; weight-reuse tiling is a follow-up).

use crate::scheme::QuantValue;
use cubecl::prelude::*;
use cubecl::server::Handle;
use half::f16;

/// Extract code `j` (6-bit) from a dense unit starting at u32 offset `base`.
#[cube]
fn dense_code(codes: &[u32], base: usize, #[comptime] j: usize) -> u32 {
    let word = comptime!((j * 6) / 32);
    let bitoff = comptime!((j * 6) % 32);
    let lo = codes[base + word] >> comptime!(bitoff as u32);
    if comptime!(bitoff + 6 > 32) {
        (lo | (codes[base + word + 1] << comptime!((32 - bitoff) as u32))) & 0x3fu32
    } else {
        lo & 0x3fu32
    }
}

#[cube(launch_unchecked)]
fn qa_matmul_tq6_kernel(
    a_codes: &[u32],
    a_scales: &[f16],
    w_codes: &[u32],
    w_scales: &[f16],
    out: &mut [f32],
    #[comptime] n: u32,
    #[comptime] k: u32,
) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        let nn = n as usize;
        let mi = pos / nn;
        let ni = pos % nn;
        let units = comptime!((k / 16) as usize); // 16 codes per dense unit
        let wpu = 3usize; // u32 per unit (lcm(6,32)/32)

        // centroid table (comptime-filled, runtime-indexed).
        let mut lut = Array::<f32>::new(64usize);
        #[unroll]
        for i in 0..64usize {
            lut[i] = f32::new(comptime!(crate::codebook::centroid(QuantValue::Q6F, i)));
        }

        let mut acc = 0.0f32;
        for u in 0..units {
            let a_base = (mi * units + u) * wpu;
            let w_base = (ni * units + u) * wpu;
            let sa = f32::cast_from(a_scales[mi * units + u]);
            let sw = f32::cast_from(w_scales[ni * units + u]);

            let mut block = 0.0f32;
            #[unroll]
            for j in 0..16usize {
                let ai = dense_code(a_codes, a_base, j);
                let wi = dense_code(w_codes, w_base, j);
                block += lut[ai as usize] * lut[wi as usize];
            }
            acc += block * sa * sw;
        }
        out[pos] = acc;
    }
}

/// Launch the TQ6 QA matmul. `a_codes`/`w_codes` are dense TQ6 (3 u32 per 16-code
/// unit); `a_scales`/`w_scales` are one fp16 scale per unit (half-block).
/// Shapes: A `[m, k]`, W `[n, k]`, out `[m, n]`. `k` divisible by 16.
#[allow(clippy::too_many_arguments)]
pub fn launch<R: Runtime>(
    client: &ComputeClient<R>,
    a_codes: Handle,
    a_scales: Handle,
    w_codes: Handle,
    w_scales: Handle,
    out: Handle,
    m: usize,
    n: usize,
    k: usize,
) {
    let units = k / 16;
    let threads = (m * n) as u32;
    unsafe {
        qa_matmul_tq6_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(threads.div_ceil(256), 1, 1),
            CubeDim::new_1d(256),
            BufferArg::from_raw_parts(a_codes, m * units * 3),
            BufferArg::from_raw_parts(a_scales, m * units),
            BufferArg::from_raw_parts(w_codes, n * units * 3),
            BufferArg::from_raw_parts(w_scales, n * units),
            BufferArg::from_raw_parts(out, m * n),
            n as u32,
            k as u32,
        );
    }
}
