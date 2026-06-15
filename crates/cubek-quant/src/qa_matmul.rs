//! QA (quant-activation) matmul for table codebooks (TQ4 / TQ6 / …): both
//! operands are codebook-quantized in randomized-Hadamard (prerot) space. This
//! is bee's `tq_projection_qa` path — the production decode forward.
//!
//! For each output `C[m,n] = Σ_k dequant(A)[m,k] · dequant(W)[n,k]`. Because both
//! operands share the centroid table, the inner product over a 16-code unit is
//! `(Σ_j centroid[aᵢⱼ]·centroid[wᵢⱼ]) · scale_a · scale_b` — i.e. the LUT-outer-product
//! structure (the centroid·centroid product is what bee's `tqNxN_lut` precomputes),
//! with the per-half-block scales factored out of the sum. The rotation cancels
//! because both operands live in the same prerot space.
//!
//! Codes are dense (`PackedU32Dense`): code `j` at bit `bits·j`, packed densely
//! across u32 words. The dense unit is 16 codes (one half-block scale), which is
//! `bits·16/32` u32 — 3 for TQ6, 2 for TQ4. This holds for any even bit width;
//! odd widths (TQ3) would need a 32-code unit and are not handled here.
//!
//! Nothing in this module bakes a codebook: the centroid table and the RHT sign
//! pattern arrive as runtime buffers (`codebook`, `rht_signs`), so a caller
//! chooses the format by passing the matching [`QuantValue`] plus its table.
//! [`upload_codebook`] / [`upload_rht_signs`] build the default buffers from
//! [`crate::codebook`]; callers may upload their own instead.

use crate::scheme::QuantValue;
use cubecl::prelude::*;
use cubecl::server::Handle;
use half::f16;

/// Extract code `j` (`bits`-wide) from a dense u32 stream starting at u32 offset
/// `base`. Codes can straddle a word boundary, so a straddling read pulls the
/// high bits from the next word. `value` is comptime, so the word index, bit
/// offset and mask all fold to constants.
#[cube]
fn dense_code(codes: &[u32], base: usize, #[comptime] j: usize, #[comptime] value: QuantValue) -> u32 {
    let bits = comptime!(value.size_bits());
    let word = comptime!((j * bits) / 32);
    let bitoff = comptime!((j * bits) % 32);
    let mask = comptime!((1u32 << bits) - 1);
    let lo = codes[base + word] >> comptime!(bitoff as u32);
    if comptime!(bitoff + bits > 32) {
        (lo | (codes[base + word + 1] << comptime!((32 - bitoff) as u32))) & mask
    } else {
        lo & mask
    }
}

/// `bits·16/32` — u32 per 16-code (half-block) dense unit. Comptime helper.
fn words_per_unit(value: QuantValue) -> usize {
    value.size_bits() * 16 / 32
}

#[cube(launch_unchecked)]
fn qa_matmul_kernel(
    a_codes: &[u32],
    a_scales: &[f16],
    w_codes: &[u32],
    w_scales: &[f16],
    codebook: &[f32],
    out: &mut [f32],
    #[comptime] n: u32,
    #[comptime] k: u32,
    #[comptime] value: QuantValue,
) {
    let pos = ABSOLUTE_POS as usize;
    if pos < out.len() {
        let nn = n as usize;
        let mi = pos / nn;
        let ni = pos % nn;
        let units = comptime!((k / 16) as usize); // 16 codes per dense unit
        let wpu = comptime!(words_per_unit(value)); // u32 per unit
        let levels = comptime!(crate::codebook::num_levels(value));

        // centroid table (loaded from the runtime buffer, runtime-indexed).
        let mut lut = Array::<f32>::new(levels);
        #[unroll]
        for i in 0..levels {
            lut[i] = codebook[i];
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
                let ai = dense_code(a_codes, a_base, j, value);
                let wi = dense_code(w_codes, w_base, j, value);
                block += lut[ai as usize] * lut[wi as usize];
            }
            acc += block * sa * sw;
        }
        out[pos] = acc;
    }
}

/// Weight-reuse tiled QA matmul: one cube per N-column dequantizes `W[col]`
/// (the whole K) into shared memory ONCE, then all threads sweep the M rows
/// against the cached column. Each weight row is dequantized exactly once
/// (vs M·N times in [`qa_matmul_kernel`]); M-parallelism is the cube's
/// thread count, which suits training-sized batches. Activations arrive
/// pre-dequantized as f32 `[M, K]` (e.g. from `dequantize::launch_ref` on the
/// codebook activation codes).
#[cube(launch_unchecked)]
fn qa_gemm_panel_kernel(
    a: &[f32],       // [M, K] dequantized (prerot-space) activations
    w_codes: &[u32], // [N, K] dense codebook weights
    w_scales: &[f16],
    codebook: &[f32],
    out: &mut [f32], // [M, N]
    #[comptime] m: u32,
    #[comptime] n: u32,
    #[comptime] k: u32,
    #[comptime] value: QuantValue,
) {
    let col = CUBE_POS_X as usize;
    if col < n as usize {
        let kk = k as usize;
        let mm = m as usize;
        let nn = n as usize;
        let units = comptime!((k / 16) as usize);
        let wpu = comptime!(words_per_unit(value));
        let levels = comptime!(crate::codebook::num_levels(value));

        let mut lut = Array::<f32>::new(levels);
        #[unroll]
        for i in 0..levels {
            lut[i] = codebook[i];
        }

        // dequant W[col] into shared, cooperatively (one unit per stride).
        let mut w_col = Shared::<[f32]>::new_slice(k as usize);
        let mut u = UNIT_POS as usize;
        while u < units {
            let w_base = (col * units + u) * wpu;
            let sw = f32::cast_from(w_scales[col * units + u]);
            #[unroll]
            for j in 0..16usize {
                let wi = dense_code(w_codes, w_base, j, value);
                w_col[u * 16 + j] = lut[wi as usize] * sw;
            }
            u += CUBE_DIM as usize;
        }
        sync_cube();

        // sweep M rows against the cached column.
        let mut row = UNIT_POS as usize;
        while row < mm {
            let mut acc = 0.0f32;
            let a_base = row * kk;
            for c in 0..kk {
                acc += a[a_base + c] * w_col[c];
            }
            out[row * nn + col] = acc;
            row += CUBE_DIM as usize;
        }
    }
}

/// Launch the weight-reuse QA matmul. `a` is f32 `[m, k]` (pre-dequantized
/// activations); weights `[n, k]` are dense codebook codes for `value`. One cube
/// per output column. `codebook` is the centroid table for `value` (length
/// `crate::codebook::num_levels(value)`).
#[allow(clippy::too_many_arguments)]
pub fn launch_panel<R: Runtime>(
    client: &ComputeClient<R>,
    value: QuantValue,
    a: Handle,
    w_codes: Handle,
    w_scales: Handle,
    codebook: Handle,
    out: Handle,
    m: usize,
    n: usize,
    k: usize,
) {
    let units = k / 16;
    let wpu = words_per_unit(value);
    let levels = crate::codebook::num_levels(value);
    unsafe {
        qa_gemm_panel_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(n as u32, 1, 1),
            CubeDim::new_1d(256),
            BufferArg::from_raw_parts(a, m * k),
            BufferArg::from_raw_parts(w_codes, n * units * wpu),
            BufferArg::from_raw_parts(w_scales, n * units),
            BufferArg::from_raw_parts(codebook, levels),
            BufferArg::from_raw_parts(out, m * n),
            m as u32,
            n as u32,
            k as u32,
            value,
        );
    }
}

/// Nearest centroid index for `val`: count of midpoint boundaries below it
/// (centroids are sorted ascending, so `count(val >= boundary[i])` is the
/// nearest index). Boundaries are the midpoints of adjacent `codebook` entries.
#[cube]
fn choose_index(val: f32, codebook: &[f32], #[comptime] value: QuantValue) -> u32 {
    let levels = comptime!(crate::codebook::num_levels(value));
    let mut idx = 0u32;
    #[unroll]
    for i in 0..(levels - 1) {
        let boundary = (codebook[i] + codebook[i + 1]) * 0.5f32;
        idx += u32::cast_from(val >= boundary);
    }
    idx
}

/// Quantize activations to a table codebook in prerot space — bee's
/// `tqN_activation_quantize` (production decode): RHT-forward → per-half-block
/// RMS seed → 6-iter Lloyd refine → nearest-centroid → dense pack. One thread
/// per 32-value block. Output layout matches [`launch`]'s `a_codes`/`a_scales`
/// (`bits` u32 + 2 fp16 per 32-value block). `codebook` and `rht_signs` are the
/// runtime-provided table and sign pattern.
#[cube(launch_unchecked)]
fn quantize_activations_kernel(
    hidden: &[f32],
    codes: &mut [u32],
    scales: &mut [f16],
    codebook: &[f32],
    rht_signs: &[f32],
    #[comptime] hidden_dim: u32,
    #[comptime] value: QuantValue,
) {
    let block = ABSOLUTE_POS as usize;
    let n_blocks = (hidden.len()) / 32;
    if block < n_blocks {
        let bpr = comptime!((hidden_dim / 32) as usize); // blocks per row
        let row = block / bpr;
        let b = block % bpr;
        let in_base = row * (hidden_dim as usize) + b * 32;
        let levels = comptime!(crate::codebook::num_levels(value));
        let words_per_block = comptime!(value.size_bits()); // bits·32/32 = bits

        // centroid table (from the runtime buffer).
        let mut lut = Array::<f32>::new(levels);
        #[unroll]
        for i in 0..levels {
            lut[i] = codebook[i];
        }

        // load + forward RHT (signs → butterfly → 1/sqrt(32)).
        let mut buf = Array::<f32>::new(32usize);
        #[unroll]
        for j in 0..32usize {
            buf[j] = hidden[in_base + j] * rht_signs[j];
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
                let c0 = lut[choose_index(buf[j] * inv_lo, codebook, value) as usize];
                num_lo += buf[j] * c0;
                den_lo += c0 * c0;
                let c1 = lut[choose_index(buf[j + 16] * inv_hi, codebook, value) as usize];
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

        // nearest-centroid + dense pack (code j at bit bits·j → `bits` u32).
        let fi_lo = if d_lo > 1e-10f32 { 1.0f32 / d_lo } else { 0.0f32 };
        let fi_hi = if d_hi > 1e-10f32 { 1.0f32 / d_hi } else { 0.0f32 };
        let mut words = Array::<u32>::new(words_per_block);
        #[unroll]
        for w in 0..words_per_block {
            words[w] = 0u32;
        }
        #[unroll]
        for j in 0..32usize {
            let inv = if j < 16 { fi_lo } else { fi_hi };
            let idx = choose_index(buf[j] * inv, codebook, value);
            let bits = comptime!(value.size_bits());
            let word = comptime!((j * bits) / 32);
            let bitoff = comptime!((j * bits) % 32);
            words[word] = words[word] | (idx << comptime!(bitoff as u32));
            if comptime!(bitoff + bits > 32) {
                words[word + 1] = words[word + 1] | (idx >> comptime!((32 - bitoff) as u32));
            }
        }
        #[unroll]
        for w in 0..words_per_block {
            codes[block * words_per_block + w] = words[w];
        }
    }
}

/// Upload the default centroid table for `value` (from [`crate::codebook`]) into
/// a device buffer, ready to pass as the `codebook` argument. Callers that own a
/// different table can build the handle themselves instead.
pub fn upload_codebook<R: Runtime>(client: &ComputeClient<R>, value: QuantValue) -> Handle {
    client.create_from_slice(f32::as_bytes(crate::codebook::table(value)))
}

/// Upload the default 32-wide RHT sign pattern into a device buffer, ready to
/// pass as the `rht_signs` argument of [`launch_activation_quant`].
pub fn upload_rht_signs<R: Runtime>(client: &ComputeClient<R>) -> Handle {
    client.create_from_slice(f32::as_bytes(&crate::codebook::RHT_SIGNS))
}

/// Quantize activations `hidden [m, k]` to the `value` table codebook (prerot
/// space). Writes dense codes `[m, k·bits/32]` u32 and `[m, k/16]` fp16 scales,
/// ready for [`launch`]. `codebook`/`rht_signs` are runtime buffers (see
/// [`upload_codebook`] / [`upload_rht_signs`]).
#[allow(clippy::too_many_arguments)]
pub fn launch_activation_quant<R: Runtime>(
    client: &ComputeClient<R>,
    value: QuantValue,
    hidden: Handle,
    codes: Handle,
    scales: Handle,
    codebook: Handle,
    rht_signs: Handle,
    m: usize,
    k: usize,
) {
    let bits = value.size_bits();
    let levels = crate::codebook::num_levels(value);
    let n_blocks = (m * k / 32) as u32;
    unsafe {
        quantize_activations_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(n_blocks.div_ceil(256), 1, 1),
            CubeDim::new_1d(256),
            BufferArg::from_raw_parts(hidden, m * k),
            BufferArg::from_raw_parts(codes, m * k * bits / 32),
            BufferArg::from_raw_parts(scales, m * k / 16),
            BufferArg::from_raw_parts(codebook, levels),
            BufferArg::from_raw_parts(rht_signs, 32),
            k as u32,
            value,
        );
    }
}

/// Launch the table-codebook QA matmul for `value`. `a_codes`/`w_codes` are
/// dense codes (`bits·16/32` u32 per 16-code unit); `a_scales`/`w_scales` are one
/// fp16 scale per unit (half-block). `codebook` is the centroid table for
/// `value`. Shapes: A `[m, k]`, W `[n, k]`, out `[m, n]`. `k` divisible by 16.
#[allow(clippy::too_many_arguments)]
pub fn launch<R: Runtime>(
    client: &ComputeClient<R>,
    value: QuantValue,
    a_codes: Handle,
    a_scales: Handle,
    w_codes: Handle,
    w_scales: Handle,
    codebook: Handle,
    out: Handle,
    m: usize,
    n: usize,
    k: usize,
) {
    let units = k / 16;
    let wpu = words_per_unit(value);
    let levels = crate::codebook::num_levels(value);
    let threads = (m * n) as u32;
    unsafe {
        qa_matmul_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(threads.div_ceil(256), 1, 1),
            CubeDim::new_1d(256),
            BufferArg::from_raw_parts(a_codes, m * units * wpu),
            BufferArg::from_raw_parts(a_scales, m * units),
            BufferArg::from_raw_parts(w_codes, n * units * wpu),
            BufferArg::from_raw_parts(w_scales, n * units),
            BufferArg::from_raw_parts(codebook, levels),
            BufferArg::from_raw_parts(out, m * n),
            n as u32,
            k as u32,
            value,
        );
    }
}
