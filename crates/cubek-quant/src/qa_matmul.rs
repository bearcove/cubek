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
//! pattern arrive as **comptime constants** ([`Codebook`], [`RhtSigns`]) injected
//! by the caller, so a caller chooses the format by passing the matching
//! [`QuantValue`] plus its table — the values are baked straight into the shader,
//! never a runtime buffer and never hardcoded in this fork. The caller (e.g. bee)
//! owns the tables.

use crate::scheme::QuantValue;
use cubecl::prelude::*;
use cubecl::server::Handle;
use core::hash::{Hash, Hasher};
use half::f16;

/// The codebook (centroid table) injected by the caller as a **comptime
/// constant** — re-exported from `cubecl-common` so cubek-quant, cubecl-std and
/// helix-burn all share the ONE `Codebook` type (no duplicate definitions). The
/// caller (e.g. bee) owns the table and passes `Codebook(&ITS_TABLE)`; `value`'s
/// `num_levels` decides how many entries are read.
pub use cubecl_common::quant::scheme::Codebook;

/// The 32-wide ±1 RHT sign pattern (the "prerot" rotation), injected by the
/// caller as a **comptime constant** — baked into the shader, never a runtime
/// buffer and never hardcoded in this fork. Same bit-pattern hash/eq as
/// [`Codebook`] so it can be a `#[comptime]` arg (the kernel cache keys on it).
#[derive(Clone, Copy, Debug)]
pub struct RhtSigns(pub &'static [f32]);

impl PartialEq for RhtSigns {
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len()
            && self
                .0
                .iter()
                .zip(other.0)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }
}
impl Eq for RhtSigns {}
impl Hash for RhtSigns {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for v in self.0 {
            v.to_bits().hash(state);
        }
    }
}

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
    out: &mut [f32],
    #[comptime] codebook: Codebook,
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

        // centroid table baked from the comptime-injected codebook.
        let mut lut = Array::<f32>::new(levels);
        #[unroll]
        for i in 0..levels {
            lut[i] = f32::new(comptime!(codebook.0[i]));
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
    a: &[f32],       // [M, K] RAW activations (forward-RHT applied in-kernel)
    w_codes: &[u32], // [N, K] dense codebook weights (rotated, packed)
    w_scales: &[f16],
    out: &mut [f32], // [M, N]
    #[comptime] codebook: Codebook,
    #[comptime] rht_signs: RhtSigns,
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
            lut[i] = f32::new(comptime!(codebook.0[i]));
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

        // sweep M rows against the cached column. A non-empty `rht_signs` selects
        // bee's `matvec_prerot`: the weight column is stored in prerot (RHT) space,
        // so we forward-RHT each 32-block of the activation row (sign → Walsh–
        // Hadamard → 1/√32) before the dot. An empty `RhtSigns(&[])` is the plain
        // panel matmul (activation already in the matching space).
        let mut row = UNIT_POS as usize;
        while row < mm {
            let mut acc = 0.0f32;
            let a_base = row * kk;
            if comptime!(rht_signs.0.len() > 0) {
                let nblk = kk / 32;
                let mut blk = 0usize;
                while blk < nblk {
                    let blk_base = a_base + blk * 32;
                    let mut buf = Array::<f32>::new(32usize);
                    #[unroll]
                    for j in 0..32usize {
                        buf[j] = a[blk_base + j] * comptime!(rht_signs.0[j]);
                    }
                    let mut step = 1usize;
                    while step < 32 {
                        let span = step * 2;
                        let mut q = 0usize;
                        while q < 32 {
                            for idx in q..q + step {
                                let lo = buf[idx];
                                let hi = buf[idx + step];
                                buf[idx] = lo + hi;
                                buf[idx + step] = lo - hi;
                            }
                            q += span;
                        }
                        step *= 2;
                    }
                    #[unroll]
                    for j in 0..32usize {
                        acc +=
                            buf[j] * comptime!(crate::codebook::INV_SQRT32) * w_col[blk * 32 + j];
                    }
                    blk += 1;
                }
            } else {
                for c in 0..kk {
                    acc += a[a_base + c] * w_col[c];
                }
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
#[allow(clippy::too_many_arguments)]
pub fn launch_panel<R: Runtime>(
    client: &ComputeClient<R>,
    value: QuantValue,
    a: Handle,
    w_codes: Handle,
    w_scales: Handle,
    codebook: Codebook,
    rht_signs: RhtSigns,
    out: Handle,
    m: usize,
    n: usize,
    k: usize,
) {
    let units = k / 16;
    let wpu = words_per_unit(value);
    unsafe {
        qa_gemm_panel_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(n as u32, 1, 1),
            CubeDim::new_1d(256),
            BufferArg::from_raw_parts(a, m * k),
            BufferArg::from_raw_parts(w_codes, n * units * wpu),
            BufferArg::from_raw_parts(w_scales, n * units),
            BufferArg::from_raw_parts(out, m * n),
            codebook,
            rht_signs,
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
fn choose_index(val: f32, #[comptime] codebook: Codebook, #[comptime] value: QuantValue) -> u32 {
    let levels = comptime!(crate::codebook::num_levels(value));
    let mut idx = 0u32;
    #[unroll]
    for i in 0..(levels - 1) {
        let boundary = comptime!((codebook.0[i] + codebook.0[i + 1]) * 0.5);
        idx += u32::cast_from(val >= boundary);
    }
    idx
}

/// Quantize activations to a table codebook in prerot space — bee's
/// `tqN_activation_quantize` (production decode): RHT-forward → per-half-block
/// RMS seed → 6-iter Lloyd refine → nearest-centroid → dense pack. One thread
/// per 32-value block. Output layout matches [`launch`]'s `a_codes`/`a_scales`
/// (`bits` u32 + 2 fp16 per 32-value block). `codebook` and `rht_signs` are the
/// comptime-injected table and sign pattern.
#[cube(launch_unchecked)]
fn quantize_activations_kernel(
    hidden: &[f32],
    codes: &mut [u32],
    scales: &mut [f16],
    #[comptime] codebook: Codebook,
    #[comptime] rht_signs: RhtSigns,
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

        // centroid table baked from the comptime-injected codebook.
        let mut lut = Array::<f32>::new(levels);
        #[unroll]
        for i in 0..levels {
            lut[i] = f32::new(comptime!(codebook.0[i]));
        }

        // load + forward RHT (signs → butterfly → 1/sqrt(32)).
        let mut buf = Array::<f32>::new(32usize);
        #[unroll]
        for j in 0..32usize {
            buf[j] = hidden[in_base + j] * comptime!(rht_signs.0[j]);
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

/// Quantize activations `hidden [m, k]` to the `value` table codebook (prerot
/// space). Writes dense codes `[m, k·bits/32]` u32 and `[m, k/16]` fp16 scales,
/// ready for [`launch`]. `codebook`/`rht_signs` are comptime-injected by the
/// caller (see [`Codebook`] / [`RhtSigns`]).
#[allow(clippy::too_many_arguments)]
pub fn launch_activation_quant<R: Runtime>(
    client: &ComputeClient<R>,
    value: QuantValue,
    hidden: Handle,
    codes: Handle,
    scales: Handle,
    codebook: Codebook,
    rht_signs: RhtSigns,
    m: usize,
    k: usize,
) {
    let bits = value.size_bits();
    let n_blocks = (m * k / 32) as u32;
    unsafe {
        quantize_activations_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(n_blocks.div_ceil(256), 1, 1),
            CubeDim::new_1d(256),
            BufferArg::from_raw_parts(hidden, m * k),
            BufferArg::from_raw_parts(codes, m * k * bits / 32),
            BufferArg::from_raw_parts(scales, m * k / 16),
            codebook,
            rht_signs,
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
    codebook: Codebook,
    out: Handle,
    m: usize,
    n: usize,
    k: usize,
) {
    let units = k / 16;
    let wpu = words_per_unit(value);
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
            BufferArg::from_raw_parts(out, m * n),
            codebook,
            n as u32,
            k as u32,
            value,
        );
    }
}
