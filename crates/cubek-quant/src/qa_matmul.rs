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

/// Nearest TQ6 centroid index for `val` (sorted centroids → count of midpoint
/// boundaries below `val`).
#[cube]
fn tq6_choose_index(val: f32) -> u32 {
    let mut idx = 0u32;
    #[unroll]
    for i in 0..63usize {
        idx += u32::cast_from(val >= comptime!(crate::codebook::boundary(QuantValue::Q6F, i)));
    }
    idx
}

/// Quantize activations to TQ6 in prerot space — bee's `tq6_activation_quantize`
/// (production decode): RHT-forward → per-half-block RMS seed → 6-iter Lloyd
/// refine → nearest-centroid → dense pack. One thread per 32-value block. Output
/// layout matches [`launch`]'s `a_codes`/`a_scales` (6 u32 + 2 fp16 per block).
#[cube(launch_unchecked)]
fn tq6_quantize_activations_kernel(
    hidden: &[f32],
    codes: &mut [u32],
    scales: &mut [f16],
    #[comptime] hidden_dim: u32,
) {
    let block = ABSOLUTE_POS as usize;
    let n_blocks = (hidden.len()) / 32;
    if block < n_blocks {
        let bpr = comptime!((hidden_dim / 32) as usize); // blocks per row
        let row = block / bpr;
        let b = block % bpr;
        let in_base = row * (hidden_dim as usize) + b * 32;

        // centroid table.
        let mut lut = Array::<f32>::new(64usize);
        #[unroll]
        for i in 0..64usize {
            lut[i] = f32::new(comptime!(crate::codebook::centroid(QuantValue::Q6F, i)));
        }

        // load + forward RHT (signs → butterfly → 1/sqrt(32)).
        let mut buf = Array::<f32>::new(32usize);
        #[unroll]
        for j in 0..32usize {
            buf[j] = hidden[in_base + j] * comptime!(crate::codebook::RHT_SIGNS[j]);
        }
        let mut step = 1usize;
        while step < 32 {
            let span = step * 2;
            let mut q = 0usize;
            while q < 32 {
                for idx in q..q + step {
                    let a = buf[idx];
                    let c = buf[idx + step];
                    buf[idx] = a + c;
                    buf[idx + step] = a - c;
                }
                q += span;
            }
            step *= 2;
        }
        #[unroll]
        for j in 0..32usize {
            buf[j] = buf[j] * comptime!(crate::codebook::INV_SQRT32);
        }

        // per-half-block RMS seed.
        let mut s_lo = 0.0f32;
        let mut s_hi = 0.0f32;
        #[unroll]
        for j in 0..16usize {
            s_lo += buf[j] * buf[j];
            s_hi += buf[j + 16] * buf[j + 16];
        }
        let mut d_lo = (s_lo / 16.0f32).sqrt();
        let mut d_hi = (s_hi / 16.0f32).sqrt();

        // 6-iter Lloyd refine: d = Σ(v·c) / Σ(c·c).
        for _iter in 0..6u32 {
            let inv_lo = if d_lo > 1e-10f32 { 1.0f32 / d_lo } else { 0.0f32 };
            let inv_hi = if d_hi > 1e-10f32 { 1.0f32 / d_hi } else { 0.0f32 };
            let mut num_lo = 0.0f32;
            let mut den_lo = 0.0f32;
            let mut num_hi = 0.0f32;
            let mut den_hi = 0.0f32;
            #[unroll]
            for j in 0..16usize {
                let c0 = lut[tq6_choose_index(buf[j] * inv_lo) as usize];
                num_lo += buf[j] * c0;
                den_lo += c0 * c0;
                let c1 = lut[tq6_choose_index(buf[j + 16] * inv_hi) as usize];
                num_hi += buf[j + 16] * c1;
                den_hi += c1 * c1;
            }
            if den_lo > 1e-10f32 {
                d_lo = num_lo / den_lo;
            }
            if den_hi > 1e-10f32 {
                d_hi = num_hi / den_hi;
            }
        }

        scales[block * 2] = f16::cast_from(d_lo);
        scales[block * 2 + 1] = f16::cast_from(d_hi);

        // nearest-centroid + dense pack (code j at bit 6j → 6 u32).
        let fi_lo = if d_lo > 1e-10f32 { 1.0f32 / d_lo } else { 0.0f32 };
        let fi_hi = if d_hi > 1e-10f32 { 1.0f32 / d_hi } else { 0.0f32 };
        let mut words = Array::<u32>::new(6usize);
        #[unroll]
        for w in 0..6usize {
            words[w] = 0u32;
        }
        #[unroll]
        for j in 0..32usize {
            let inv = if j < 16 { fi_lo } else { fi_hi };
            let idx = tq6_choose_index(buf[j] * inv);
            let word = comptime!((j * 6) / 32);
            let bitoff = comptime!((j * 6) % 32);
            words[word] = words[word] | (idx << comptime!(bitoff as u32));
            if comptime!(bitoff + 6 > 32) {
                words[word + 1] = words[word + 1] | (idx >> comptime!((32 - bitoff) as u32));
            }
        }
        #[unroll]
        for w in 0..6usize {
            codes[block * 6 + w] = words[w];
        }
    }
}

/// Quantize activations `hidden [m, k]` to TQ6 (prerot space). Writes dense
/// codes `[m, k*6/32]` u32 and `[m, k/16]` fp16 scales, ready for [`launch`].
pub fn launch_activation_quant<R: Runtime>(
    client: &ComputeClient<R>,
    hidden: Handle,
    codes: Handle,
    scales: Handle,
    m: usize,
    k: usize,
) {
    let n_blocks = (m * k / 32) as u32;
    unsafe {
        tq6_quantize_activations_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(n_blocks.div_ceil(256), 1, 1),
            CubeDim::new_1d(256),
            BufferArg::from_raw_parts(hidden, m * k),
            BufferArg::from_raw_parts(codes, m * k * 6 / 32),
            BufferArg::from_raw_parts(scales, m * k / 16),
            k as u32,
        );
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
