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

/// Runtime-`j` variant of [`dense_code`]: extract code `j` (0-based within a
/// 16-code unit) when `j` comes from a runtime value (e.g. the warp lane in the
/// GEMV kernel). `bits` is comptime so the mask folds, but the word index, bit
/// offset and the straddle test are runtime.
#[cube]
fn dense_code_dyn(codes: &[u32], base: usize, j: u32, #[comptime] value: QuantValue) -> u32 {
    let bits = comptime!(value.size_bits() as u32);
    let mask = comptime!((1u32 << value.size_bits()) - 1);
    let bitpos = j * bits;
    let word = (bitpos / 32) as usize;
    let off = bitpos % 32;
    let lo = codes[base + word] >> off;
    if off + bits > 32 {
        (lo | (codes[base + word + 1] << (32u32 - off))) & mask
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

/// Weight-reuse QA matmul: one cube per N-column dequantizes `W[col]` (the whole
/// K) into shared memory ONCE, then the WHOLE cube cooperates on each output
/// `out[row,col] = a[row]·W[col]`. Activations arrive RAW `[M, K]` (f16/f32) and
/// the forward-RHT (prerot) is applied in-kernel.
///
/// Two measured fixes over the original one-thread-per-row panel (which left
/// 255/256 threads idle for the M=1 decode shape — occupancy 17%, latency-bound):
///   1. **Codebook LUT lives in shared memory**, not a per-thread `Array`. A
///      dynamically-indexed local array spills to local memory, so every centroid
///      lookup was a scattered L2 load (the ncu bottleneck). One shared copy per
///      cube keeps all lookups on-chip.
///   2. **The K-reduction is cube-cooperative**: every row's dot is split across
///      all 256 threads and reduced (warp `plane_sum` → shared scratch → thread
///      0). For the RHT path the per-32-block Walsh–Hadamard is done by one WARP
///      (lane = element, butterfly via `plane_shuffle_xor`) — no local `buf[32]`,
///      so the whole prerot+dot stays in registers + shared.
#[cube(launch_unchecked)]
fn qa_gemm_panel_kernel<F: Float>(
    a: &[F],         // [M, K] RAW activations (f16/f32), forward-RHT applied in-kernel
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
    // One cube per output column. `n` can exceed the 65535 grid-dim cap (e.g. the
    // LM head, n = vocab), so columns are laid out across a 2D grid.
    let col = (CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X) as usize;
    if col < n as usize {
        let kk = k as usize;
        let mm = m as usize;
        let nn = n as usize;
        let units = comptime!((k / 16) as usize);
        let wpu = comptime!(words_per_unit(value));
        let levels = comptime!(crate::codebook::num_levels(value));

        // --- shared codebook LUT (one copy per cube; thread 0 bakes constants) ---
        let mut lut = Shared::<[f32]>::new_slice(levels);
        // --- shared ±1 RHT sign pattern (32-wide), indexable by runtime lane ---
        let mut signs = Shared::<[f32]>::new_slice(32usize);
        if UNIT_POS == 0 {
            if comptime!(!value.is_symmetric()) {
                #[unroll]
                for i in 0..levels {
                    lut[i] = f32::new(comptime!(codebook.0[i]));
                }
            }
            if comptime!(rht_signs.0.len() > 0) {
                #[unroll]
                for i in 0..32usize {
                    signs[i] = f32::new(comptime!(rht_signs.0[i]));
                }
            }
        }
        sync_cube();

        // dequant W[col] into shared, cooperatively (one unit per stride).
        let mut w_col = Shared::<[f32]>::new_slice(k as usize);
        let mut u = UNIT_POS as usize;
        while u < units {
            let w_base = (col * units + u) * wpu;
            let sw = f32::cast_from(w_scales[col * units + u]);
            #[unroll]
            for j in 0..16usize {
                let raw = dense_code(w_codes, w_base, j, value);
                let wv = if comptime!(value.is_symmetric()) {
                    let sign_bit = comptime!(1u32 << (value.size_bits() - 1));
                    let two_pow = comptime!(1i32 << value.size_bits());
                    let neg = i32::cast_from(raw >= sign_bit);
                    f32::cast_from(i32::cast_from(raw) - neg * two_pow)
                } else {
                    lut[raw as usize]
                };
                w_col[u * 16 + j] = wv * sw;
            }
            u += CUBE_DIM as usize;
        }
        sync_cube();

        // cross-warp reduction scratch (one slot per warp; ≤ 256/min-plane).
        let mut scratch = Shared::<[f32]>::new_slice(32usize);
        let lane = UNIT_POS_PLANE;
        let warp = UNIT_POS / PLANE_DIM;
        let n_warps = CUBE_DIM / PLANE_DIM;

        // sweep M rows against the cached column, the whole cube cooperating per
        // row. A non-empty `rht_signs` selects bee's `matvec_prerot`: the weight
        // column is stored in prerot (RHT) space, so each 32-block of the
        // activation row is forward-RHT'd (sign → Walsh–Hadamard → 1/√32) before
        // the dot. An empty `RhtSigns(&[])` is the plain panel matmul.
        for row in 0..mm {
            let a_base = row * kk;
            let mut acc = 0.0f32;
            if comptime!(rht_signs.0.len() > 0) {
                // each WARP owns whole 32-blocks (lane = element within the block).
                let nblk = kk / 32;
                let mut blk = warp as usize;
                while blk < nblk {
                    let blk_base = a_base + blk * 32;
                    let mut v = f32::cast_from(a[blk_base + lane as usize]) * signs[lane as usize];
                    // length-32 Walsh–Hadamard butterfly across the warp lanes:
                    // pair (l, l^mask); low lane → v+p, high lane → p-v.
                    let mut mask = 1u32;
                    while mask < 32 {
                        let p = plane_shuffle_xor(v, mask);
                        if (lane & mask) == 0 {
                            v = v + p;
                        } else {
                            v = p - v;
                        }
                        mask *= 2;
                    }
                    acc += v
                        * comptime!(crate::codebook::INV_SQRT32)
                        * w_col[blk * 32 + lane as usize];
                    blk += n_warps as usize;
                }
            } else {
                // plain dot: each thread strides over K, reduced below.
                let mut c = UNIT_POS as usize;
                while c < kk {
                    acc += f32::cast_from(a[a_base + c]) * w_col[c];
                    c += CUBE_DIM as usize;
                }
            }

            // warp-reduce, then cross-warp via shared scratch, then thread 0 writes.
            let warp_total = plane_sum(acc);
            if lane == 0 {
                scratch[warp as usize] = warp_total;
            }
            sync_cube();
            if UNIT_POS == 0 {
                let mut tot = 0.0f32;
                for w in 0..n_warps as usize {
                    tot += scratch[w];
                }
                out[row * nn + col] = tot;
            }
            sync_cube();
        }
    }
}

/// Decode GEMV (M = 1): **one warp per output column** — bee's Metal
/// `tq6_1s_matvec_prerot` shape. A cube holds `CUBE_DIM/PLANE_DIM` warps and
/// handles that many columns; lane = element within the 32-block, the K-dot is a
/// single `plane_sum` per warp, and the weight is dequantized straight on read
/// (no `w_col` staging — there is no row reuse at M=1, so staging would only add
/// a shared round-trip, which `qa_gemm_panel_kernel` showed becomes the L1/TEX
/// bottleneck). The forward-RHT'd activation is staged ONCE into shared per cube
/// and reused across the cube's columns (the `tile_wide` trick), so the butterfly
/// runs `CUBE_DIM/PLANE_DIM`× fewer times than a per-column in-line RHT.
#[cube(launch_unchecked)]
fn qa_gemv_kernel<F: Float, S: Float>(
    a: &[F],         // [1, K] RAW activation (RMSNorm + forward-RHT applied in-kernel)
    w_codes: &[u32], // [N, K] dense codebook weights (rotated, packed)
    w_scales: &[f16],
    out: &mut [f32], // [1, N]
    gamma: &[F],     // [K] RMSNorm gamma (only read when do_norm; dummy otherwise)
    #[comptime] codebook: Codebook,
    #[comptime] rht_signs: RhtSigns,
    #[comptime] n: u32,
    #[comptime] k: u32,
    #[comptime] value: QuantValue,
    #[comptime] do_norm: bool, // fold input_ln/post_ln RMSNorm into the gemv
) {
    let kk = k as usize;
    let nn = n as usize;
    let units = comptime!((k / 16) as usize);
    let wpu = comptime!(words_per_unit(value));
    let levels = comptime!(crate::codebook::num_levels(value));
    let nblk = comptime!((k / 32) as usize);

    let lane = UNIT_POS_PLANE;
    let warp = UNIT_POS / PLANE_DIM;
    let n_warps = CUBE_DIM / PLANE_DIM;
    let col = ((CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X) * n_warps + warp) as usize;

    // shared LUT, shared ±1 signs, the staged (RHT'd) activation row, and per-warp
    // scratch for the in-kernel RMSNorm reduction.
    let mut lut = Shared::<[f32]>::new_slice(levels);
    let mut signs = Shared::<[f32]>::new_slice(32usize);
    // Staged (post-RHT) activation in `S` (f16 by default, f32 via QA_STAGE_F32):
    // this is the dominant threadgroup allocation (K·4B in f32), and halving it to
    // K·2B roughly doubles cube co-residency / occupancy. The butterfly
    // accumulation and the Σh² reduction stay f32 (the reduction uses the f32
    // register `hi`, not a_sh), so only the staged value rounds to f16 —
    // negligible against 4-bit weights.
    let mut a_sh = Shared::<[S]>::new_slice(k as usize);
    let mut ss_sh = Shared::<[f32]>::new_slice(32usize);
    if UNIT_POS == 0 {
        // Codebook (table) values fill the LUT; symmetric (Q4S/Q8S/…) read no
        // centroid table — the dequant is `signed(raw)·scale`, so skip the fill
        // (codebook.0 is empty for symmetric values).
        if comptime!(!value.is_symmetric()) {
            #[unroll]
            for i in 0..levels {
                lut[i] = f32::new(comptime!(codebook.0[i]));
            }
        }
        if comptime!(rht_signs.0.len() > 0) {
            #[unroll]
            for i in 0..32usize {
                signs[i] = f32::new(comptime!(rht_signs.0[i]));
            }
        }
    }
    sync_cube();

    // Per-row RMSNorm scale s = rsqrt(mean(h²)+eps); applied to the output column at
    // the end since it factors out of the linear dot. 1.0 when !do_norm.
    let mut s = 1.0f32;
    if comptime!(do_norm) {
        // Pass 1: accumulate Σh² and stage h⊙gamma into a_sh.
        let mut ss = 0.0f32;
        let mut i = UNIT_POS as usize;
        while i < kk {
            let hi = f32::cast_from(a[i]);
            ss += hi * hi;
            a_sh[i] = S::cast_from(hi * f32::cast_from(gamma[i]));
            i += CUBE_DIM as usize;
        }
        // Cross-warp reduce Σh²: warp plane_sum → per-warp shared → thread 0 sums.
        let wsum = plane_sum(ss);
        if lane == 0 {
            ss_sh[warp as usize] = wsum;
        }
        sync_cube();
        if UNIT_POS == 0 {
            let mut tot = 0.0f32;
            let mut w = 0usize;
            while w < n_warps as usize {
                tot += ss_sh[w];
                w += 1;
            }
            ss_sh[0] = tot / (k as f32) + 1.0e-6f32; // mean-square + eps (reuse slot 0)
        }
        sync_cube();
        s = ss_sh[0].sqrt().recip();
        sync_cube();
        // RHT in place on the gamma'd activation (per 32-block butterfly).
        if comptime!(rht_signs.0.len() > 0) {
            let mut blk = warp as usize;
            while blk < nblk {
                let mut v = f32::cast_from(a_sh[blk * 32 + lane as usize]) * signs[lane as usize];
                let mut mask = 1u32;
                while mask < 32 {
                    let p = plane_shuffle_xor(v, mask);
                    if (lane & mask) == 0 { v = v + p; } else { v = p - v; }
                    mask *= 2;
                }
                a_sh[blk * 32 + lane as usize] = S::cast_from(v * comptime!(crate::codebook::INV_SQRT32));
                blk += n_warps as usize;
            }
            sync_cube();
        }
    } else if comptime!(rht_signs.0.len() > 0) {
        // stage with forward-RHT per 32-block (warp = block, lane = element).
        let mut blk = warp as usize;
        while blk < nblk {
            let mut v = f32::cast_from(a[blk * 32 + lane as usize]) * signs[lane as usize];
            let mut mask = 1u32;
            while mask < 32 {
                let p = plane_shuffle_xor(v, mask);
                if (lane & mask) == 0 { v = v + p; } else { v = p - v; }
                mask *= 2;
            }
            a_sh[blk * 32 + lane as usize] = S::cast_from(v * comptime!(crate::codebook::INV_SQRT32));
            blk += n_warps as usize;
        }
        sync_cube();
    } else {
        let mut i = UNIT_POS as usize;
        while i < kk {
            a_sh[i] = S::cast_from(a[i]);
            i += CUBE_DIM as usize;
        }
        sync_cube();
    }

    // each warp dots its column against the staged activation, dequant-on-read.
    if col < nn {
        let mut partial = 0.0f32;
        let mut blk = 0usize;
        while blk < nblk {
            let p = blk * 32 + lane as usize; // element index in K
            let unit = p / 16; // half-block unit (own fp16 scale)
            let jj = (p % 16) as u32; // code index within the unit
            let base = (col * units + unit) * wpu;
            let raw = dense_code_dyn(w_codes, base, jj, value);
            // Codebook → centroid gather; symmetric → two's-complement signed
            // value (no LUT/gather), the cheap affine path (the iPhone-bound dequant).
            let wq = if comptime!(value.is_symmetric()) {
                let sign_bit = comptime!(1u32 << (value.size_bits() - 1));
                let two_pow = comptime!(1i32 << value.size_bits());
                let neg = i32::cast_from(raw >= sign_bit);
                f32::cast_from(i32::cast_from(raw) - neg * two_pow)
            } else {
                lut[raw as usize]
            };
            let sw = f32::cast_from(w_scales[col * units + unit]);
            partial += wq * sw * f32::cast_from(a_sh[p]);
            blk += 1;
        }
        let total = plane_sum(partial);
        if lane == 0 {
            out[col] = total * s; // s = RMSNorm scale (1.0 when !do_norm)
        }
    }
}

/// Launch the weight-reuse QA matmul. `a` is f32 `[m, k]` (pre-dequantized
/// activations); weights `[n, k]` are dense codebook codes for `value`. One cube
/// per output column. `codebook` is the centroid table for `value` (length
/// `crate::codebook::num_levels(value)`).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub fn launch_panel<R: Runtime, F: Float>(
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
    // RMSNorm fold (decode m==1 only): `gamma` is `[k]`; `do_norm` enables the
    // in-kernel input_ln/post_ln. When `do_norm` is false `gamma` is unread (pass a
    // dummy handle). `eps` is the RMSNorm epsilon.
    gamma: Handle,
    do_norm: bool,
    _eps: f32, // RMSNorm eps is the fixed model 1e-6, hardcoded in the kernel
) {
    debug_assert!(!do_norm || m == 1, "in-kernel RMSNorm fold only supports the m==1 gemv");
    let units = k / 16;
    let wpu = words_per_unit(value);
    // Decode (M = 1): the warp-per-column GEMV — no row reuse, so dequant-on-read
    // beats the panel's w_col staging. A cube of 256 threads = 8 warps = 8 cols.
    if m == 1 {
        const WARPS: usize = 8; // CubeDim 256 / warp 32 (CUDA)
        let cubes = n.div_ceil(WARPS);
        let grid_x = cubes.min(65535);
        let grid_y = cubes.div_ceil(grid_x);
        // Staged activation lives in threadgroup memory (the dominant alloc): f16
        // by default to lift occupancy; QA_STAGE_F32=1 forces the f32 baseline for
        // A/B (occupancy + WER) measurement.
        let stage_f32 = std::env::var("QA_STAGE_F32").is_ok();
        macro_rules! gemv {
            ($s:ty) => {
                qa_gemv_kernel::launch_unchecked::<F, $s, R>(
                    client,
                    CubeCount::Static(grid_x as u32, grid_y as u32, 1),
                    CubeDim::new_1d(256),
                    BufferArg::from_raw_parts(a, m * k),
                    BufferArg::from_raw_parts(w_codes, n * units * wpu),
                    BufferArg::from_raw_parts(w_scales, n * units),
                    BufferArg::from_raw_parts(out, m * n),
                    BufferArg::from_raw_parts(gamma, if do_norm { k } else { 1 }),
                    codebook,
                    rht_signs,
                    n as u32,
                    k as u32,
                    value,
                    do_norm,
                )
            };
        }
        unsafe {
            if stage_f32 {
                gemv!(f32);
            } else {
                gemv!(f16);
            }
        }
        return;
    }
    // Lay `n` output columns across a 2D grid: each grid dim is capped at 65535.
    let grid_x = n.min(65535);
    let grid_y = n.div_ceil(grid_x);
    unsafe {
        qa_gemm_panel_kernel::launch_unchecked::<F, R>(
            client,
            CubeCount::Static(grid_x as u32, grid_y as u32, 1),
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

/// Symmetric (affine int) analogue of [`quantize_activations_kernel`]: RHT-forward
/// → per-half-block `maxabs/qmax` scale → round-to-signed (two's complement) →
/// dense pack. No codebook/Lloyd — the cheap quant the iPhone CPU port targets.
/// Output layout is identical (dense `bits` u32 + 2 fp16 per 32-value block), so
/// the same gemv consumes it (via its `is_symmetric` branch).
#[cube(launch_unchecked)]
fn quantize_symmetric_activations_kernel(
    hidden: &[f32],
    codes: &mut [u32],
    scales: &mut [f16],
    #[comptime] rht_signs: RhtSigns,
    #[comptime] hidden_dim: u32,
    #[comptime] value: QuantValue,
) {
    let block = ABSOLUTE_POS as usize;
    let n_blocks = (hidden.len()) / 32;
    if block < n_blocks {
        let bpr = comptime!((hidden_dim / 32) as usize);
        let row = block / bpr;
        let b = block % bpr;
        let in_base = row * (hidden_dim as usize) + b * 32;
        let words_per_block = comptime!(value.size_bits());
        let qmax = comptime!(value.range().1); // 7.0 for Q4S (symmetric)
        let cmask = comptime!((1u32 << value.size_bits()) - 1);

        // load + forward RHT (signs → butterfly → 1/sqrt(32)) — identical to the
        // codebook path so the rotation matches the gemv's in-kernel RHT.
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

        // per-half-block symmetric scale = maxabs / qmax.
        let mut m_lo = 0.0f32;
        let mut m_hi = 0.0f32;
        #[unroll]
        for j in 0..16usize {
            let alo = buf[j].abs();
            let ahi = buf[j + 16].abs();
            if alo > m_lo {
                m_lo = alo;
            }
            if ahi > m_hi {
                m_hi = ahi;
            }
        }
        let d_lo = m_lo / qmax;
        let d_hi = m_hi / qmax;
        scales[block * 2] = f16::cast_from(d_lo);
        scales[block * 2 + 1] = f16::cast_from(d_hi);

        // round-to-signed + dense pack (two's complement nibble; the gemv sign-
        // extends it back).
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
            let mut q = (buf[j] * inv).round();
            if q > qmax {
                q = qmax;
            }
            if q < -qmax {
                q = -qmax;
            }
            let code = u32::cast_from(i32::cast_from(q)) & cmask;
            let bits = comptime!(value.size_bits());
            let word = comptime!((j * bits) / 32);
            let bitoff = comptime!((j * bits) % 32);
            words[word] = words[word] | (code << comptime!(bitoff as u32));
            if comptime!(bitoff + bits > 32) {
                words[word + 1] = words[word + 1] | (code >> comptime!((32 - bitoff) as u32));
            }
        }
        #[unroll]
        for w in 0..words_per_block {
            codes[block * words_per_block + w] = words[w];
        }
    }
}

/// Quantize `hidden [m, k]` to a symmetric affine `value` (Q4S/…) in prerot space:
/// dense codes `[m, k·bits/32]` u32 + `[m, k/16]` fp16 scales. Mirror of
/// [`launch_activation_quant`] for the no-codebook path.
pub fn launch_symmetric_activation_quant<R: Runtime>(
    client: &ComputeClient<R>,
    value: QuantValue,
    hidden: Handle,
    codes: Handle,
    scales: Handle,
    rht_signs: RhtSigns,
    m: usize,
    k: usize,
) {
    let bits = value.size_bits();
    let n_blocks = (m * k / 32) as u32;
    unsafe {
        quantize_symmetric_activations_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(n_blocks.div_ceil(256), 1, 1),
            CubeDim::new_1d(256),
            BufferArg::from_raw_parts(hidden, m * k),
            BufferArg::from_raw_parts(codes, m * k * bits / 32),
            BufferArg::from_raw_parts(scales, m * k / 16),
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
