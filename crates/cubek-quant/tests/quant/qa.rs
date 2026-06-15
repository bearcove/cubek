use cubecl::prelude::*;
use cubecl::{Runtime, TestRuntime};
use half::f16;

// 64-level Lloyd-Max TQ6 centroids (must match cubek-quant's table).
const Q6F: [f32; 64] = [
    -3.73971331, -3.23553866, -2.91215583, -2.66675206, -2.46556925, -2.29307792, -2.14077946,
    -2.00348979, -1.87780041, -1.76134301, -1.65240050, -1.54968499, -1.45220328, -1.35917132,
    -1.26995767, -1.18404491, -1.10100239, -1.02046671, -0.94212725, -0.86571539, -0.79099622,
    -0.71776211, -0.64582771, -0.57502585, -0.50520434, -0.43622321, -0.36795256, -0.30027058,
    -0.23306199, -0.16621658, -0.09962796, -0.03319237, 0.03319237, 0.09962796, 0.16621658,
    0.23306199, 0.30027058, 0.36795256, 0.43622321, 0.50520434, 0.57502585, 0.64582771, 0.71776211,
    0.79099622, 0.86571539, 0.94212725, 1.02046671, 1.10100239, 1.18404491, 1.26995767, 1.35917132,
    1.45220328, 1.54968499, 1.65240050, 1.76134301, 1.87780041, 2.00348979, 2.14077946, 2.29307792,
    2.46556925, 2.66675206, 2.91215583, 3.23553866, 3.73971331,
];

/// Dense-pack a row of 6-bit codes (code j at bit 6j) into u32 words.
fn pack_row(codes: &[u8]) -> Vec<u32> {
    let words = codes.len() * 6 / 32;
    let mut out = vec![0u32; words];
    for (j, &c) in codes.iter().enumerate() {
        let code = (c & 0x3f) as u32;
        let (word, bitoff) = (j * 6 / 32, (j * 6) % 32);
        out[word] |= code << bitoff;
        if bitoff + 6 > 32 {
            out[word + 1] |= code >> (32 - bitoff);
        }
    }
    out
}

// ── Independent host port of bee's production no-search TQ6 activation quant ──
// (rust/helix-metal/.../tq_activation_quant.metal `tq6_activation_quantize`).
// Uses bee's exact constants (verified identical to cubek's).
const RHT_SIGNS: [f32; 32] = [
    1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0,
    -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0,
];
const INV_SQRT32: f32 = 0.176_776_69;

fn bee_choose_index(v: f32) -> usize {
    // partition on midpoint boundaries (sorted centroids)
    let mut idx = 0usize;
    for i in 0..63 {
        if v >= (Q6F[i] + Q6F[i + 1]) * 0.5 {
            idx += 1;
        }
    }
    idx
}

fn bee_rht_forward(buf: &mut [f32; 32]) {
    for (j, x) in buf.iter_mut().enumerate() {
        *x *= RHT_SIGNS[j];
    }
    let mut step = 1;
    while step < 32 {
        let span = step * 2;
        let mut q = 0;
        while q < 32 {
            for i in q..q + step {
                let (a, b) = (buf[i], buf[i + step]);
                buf[i] = a + b;
                buf[i + step] = a - b;
            }
            q += span;
        }
        step *= 2;
    }
    for x in buf.iter_mut() {
        *x *= INV_SQRT32;
    }
}

/// Returns (32 codes, fp16-rounded d_lo, fp16-rounded d_hi) for one 32-block.
fn bee_quantize_block(hidden_block: &[f32]) -> ([u8; 32], f32, f32) {
    let mut buf = [0.0f32; 32];
    buf.copy_from_slice(hidden_block);
    bee_rht_forward(&mut buf);

    let rms = |s: &[f32]| (s.iter().map(|x| x * x).sum::<f32>() / 16.0).sqrt();
    let (mut d_lo, mut d_hi) = (rms(&buf[..16]), rms(&buf[16..]));
    for _ in 0..6 {
        let inv_lo = if d_lo > 1e-10 { 1.0 / d_lo } else { 0.0 };
        let inv_hi = if d_hi > 1e-10 { 1.0 / d_hi } else { 0.0 };
        let (mut nl, mut dl, mut nh, mut dh) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for j in 0..16 {
            let c0 = Q6F[bee_choose_index(buf[j] * inv_lo)];
            nl += buf[j] * c0;
            dl += c0 * c0;
            let c1 = Q6F[bee_choose_index(buf[j + 16] * inv_hi)];
            nh += buf[j + 16] * c1;
            dh += c1 * c1;
        }
        if dl > 1e-10 {
            d_lo = nl / dl;
        }
        if dh > 1e-10 {
            d_hi = nh / dh;
        }
    }
    // final indices use the f32 scale; stored scale is fp16 (bee's behaviour).
    let fi_lo = if d_lo > 1e-10 { 1.0 / d_lo } else { 0.0 };
    let fi_hi = if d_hi > 1e-10 { 1.0 / d_hi } else { 0.0 };
    let mut codes = [0u8; 32];
    for j in 0..32 {
        let inv = if j < 16 { fi_lo } else { fi_hi };
        codes[j] = bee_choose_index(buf[j] * inv) as u8;
    }
    (
        codes,
        f16::from_f32(d_lo).to_f32(),
        f16::from_f32(d_hi).to_f32(),
    )
}

/// PARITY: my cubek GPU QA forward vs an independent host port of bee's exact
/// production algorithm (same constants, same no-search steps). If these agree,
/// the kernels are faithful to bee's numerics — the kernel-level parity gate.
#[test]
fn test_qa_forward_parity_vs_bee() {
    let (m, n, k) = (6usize, 8usize, 128usize);
    let units = k / 16;
    let blocks_per_row = k / 32;
    let client = TestRuntime::client(&Default::default());

    let mut s = 0x5151_2727_9393u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let hidden: Vec<f32> = (0..m * k)
        .map(|_| ((next() % 4000) as f32 / 1000.0) - 2.0)
        .collect();
    let w_code: Vec<u8> = (0..n * k).map(|_| (next() % 64) as u8).collect();
    let w_scale: Vec<f16> = (0..n * units)
        .map(|_| f16::from_f32(0.3 + (next() % 1000) as f32 / 2000.0))
        .collect();
    let mut w_words = Vec::<u32>::new();
    for r in 0..n {
        w_words.extend(pack_row(&w_code[r * k..(r + 1) * k]));
    }

    // GPU: full forward.
    let hh = client.create_from_slice(f32::as_bytes(&hidden));
    let a_codes_h = client.empty(m * k * 6 / 32 * 4);
    let a_scales_h = client.empty(m * units * 2);
    cubek_quant::qa_matmul::launch_activation_quant::<TestRuntime>(
        &client, hh, a_codes_h.clone(), a_scales_h.clone(), m, k,
    );
    let wh = client.create_from_slice(u32::as_bytes(&w_words));
    let wsh = client.create_from_slice(f16::as_bytes(&w_scale));
    let outh = client.empty(m * n * 4);
    cubek_quant::qa_matmul::launch::<TestRuntime>(
        &client, a_codes_h, a_scales_h, wh, wsh, outh.clone(), m, n, k,
    );
    let c_gpu = f32::from_bytes(&client.read_one(outh).unwrap()).to_vec();

    // Host: bee's algorithm — quantize each block, then dequant-and-matmul.
    let mut a_deq = vec![0.0f32; m * k];
    for mi in 0..m {
        for b in 0..blocks_per_row {
            let base = mi * k + b * 32;
            let (codes, d_lo, d_hi) = bee_quantize_block(&hidden[base..base + 32]);
            for j in 0..32 {
                let sc = if j < 16 { d_lo } else { d_hi };
                a_deq[base + j] = Q6F[codes[j] as usize] * sc;
            }
        }
    }
    let w_deq = |ni: usize| -> Vec<f32> {
        let wrow = &w_words[ni * (k * 6 / 32)..(ni + 1) * (k * 6 / 32)];
        (0..k)
            .map(|c| {
                let sc = w_scale[ni * units + c / 16].to_f32();
                Q6F[unpack_code(wrow, c) as usize] * sc
            })
            .collect()
    };
    let mut max_rel = 0.0f32;
    let mut cmax = 0.0f32;
    for mi in 0..m {
        for ni in 0..n {
            let wd = w_deq(ni);
            let r: f32 = (0..k).map(|c| a_deq[mi * k + c] * wd[c]).sum();
            max_rel = max_rel.max((c_gpu[mi * n + ni] - r).abs());
            cmax = cmax.max(r.abs());
        }
    }
    // f32 sum-order + boundary-flip noise; bug (wrong signs/index/pack) would be ~O(|C|).
    assert!(
        max_rel <= 1e-2 * (1.0 + cmax),
        "QA parity vs bee: max|gpu-bee|={max_rel} (|C|max {cmax})"
    );
}

/// Unpack 6-bit code `c` from a row's dense-packed u32 words.
fn unpack_code(words: &[u32], c: usize) -> u8 {
    let (word, bitoff) = (c * 6 / 32, (c * 6) % 32);
    let lo = words[word] >> bitoff;
    let raw = if bitoff + 6 > 32 {
        lo | (words[word + 1] << (32 - bitoff))
    } else {
        lo
    };
    (raw & 0x3f) as u8
}

/// End-to-end QA forward: hidden → activation-quant (RHT+refine+pack) → QA matmul.
/// Validated against a host dequant-and-matmul of the *same* (read-back) quantized
/// operands — proves the two kernels chain correctly (codes/scales layout agrees).
#[test]
fn test_qa_forward_end_to_end() {
    let (m, n, k) = (4usize, 8usize, 64usize);
    let units = k / 16;
    let blocks_per_row = k / 32;
    let client = TestRuntime::client(&Default::default());

    let mut s = 0xABCDEF12_3456u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    // ~unit-variance hidden so the rotated values match the centroid distribution.
    let hidden: Vec<f32> = (0..m * k)
        .map(|_| (((next() % 2000) as f32 / 1000.0) - 1.0) * 1.5)
        .collect();
    let w_code: Vec<u8> = (0..n * k).map(|_| (next() % 64) as u8).collect();
    let w_scale: Vec<f16> = (0..n * units)
        .map(|_| f16::from_f32(0.2 + (next() % 1000) as f32 / 2000.0))
        .collect();
    let mut w_words = Vec::<u32>::new();
    for r in 0..n {
        w_words.extend(pack_row(&w_code[r * k..(r + 1) * k]));
    }

    // GPU: quantize activations, then QA matmul.
    let hh = client.create_from_slice(f32::as_bytes(&hidden));
    let a_codes_h = client.empty(m * k * 6 / 32 * 4);
    let a_scales_h = client.empty(m * units * 2);
    cubek_quant::qa_matmul::launch_activation_quant::<TestRuntime>(
        &client,
        hh,
        a_codes_h.clone(),
        a_scales_h.clone(),
        m,
        k,
    );
    let wh = client.create_from_slice(u32::as_bytes(&w_words));
    let wsh = client.create_from_slice(f16::as_bytes(&w_scale));
    let outh = client.empty(m * n * 4);
    cubek_quant::qa_matmul::launch::<TestRuntime>(
        &client,
        a_codes_h.clone(),
        a_scales_h.clone(),
        wh,
        wsh,
        outh.clone(),
        m,
        n,
        k,
    );
    let got = f32::from_bytes(&client.read_one(outh).unwrap()).to_vec();

    // read back the GPU-quantized activations and dequant on host.
    let a_words = u32::from_bytes(&client.read_one(a_codes_h).unwrap()).to_vec();
    let a_scale = f16::from_bytes(&client.read_one(a_scales_h).unwrap()).to_vec();
    let words_per_row = k * 6 / 32;
    let dequant = |words: &[u32], scales: &[f16], row: usize| -> Vec<f32> {
        let wrow = &words[row * words_per_row..(row + 1) * words_per_row];
        (0..k)
            .map(|c| {
                let block = c / 32;
                let half = c % 32 / 16; // 0 = d_lo, 1 = d_hi
                let sc = scales[row * units + block * 2 + half].to_f32();
                Q6F[unpack_code(wrow, c) as usize] * sc
            })
            .collect()
    };

    let mut max_abs = 0.0f32;
    let mut cmax = 0.0f32;
    for mi in 0..m {
        let a_deq = dequant(&a_words, &a_scale, mi);
        for ni in 0..n {
            let w_deq = dequant(&w_words, &w_scale, ni);
            let r: f32 = (0..k).map(|c| a_deq[c] * w_deq[c]).sum();
            max_abs = max_abs.max((got[mi * n + ni] - r).abs());
            cmax = cmax.max(r.abs());
        }
    }
    assert!(
        max_abs <= 1e-3 * (1.0 + cmax),
        "QA forward chain: max|gpu-host|={max_abs} (|C|max {cmax})"
    );
    // sanity: the quantizer produced non-trivial output.
    assert!(blocks_per_row >= 1 && got.iter().any(|&x| x.abs() > 1e-6));
}

/// TQ6 QA matmul: both operands TQ6-quantized; C = dequant(A) @ dequant(W)^T.
/// Validated against a CPU dequant-and-dot oracle.
#[test]
fn test_qa_matmul_tq6() {
    let (m, n, k) = (8usize, 8usize, 64usize); // k divisible by 16
    let units = k / 16;
    let client = TestRuntime::client(&Default::default());

    // deterministic pseudo-random codes + scales
    let mut s = 0xC0FFEE_1234_5678u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let a_code: Vec<u8> = (0..m * k).map(|_| (next() % 64) as u8).collect();
    let w_code: Vec<u8> = (0..n * k).map(|_| (next() % 64) as u8).collect();
    let a_scale: Vec<f16> = (0..m * units)
        .map(|_| f16::from_f32(0.2 + (next() % 1000) as f32 / 2000.0))
        .collect();
    let w_scale: Vec<f16> = (0..n * units)
        .map(|_| f16::from_f32(0.2 + (next() % 1000) as f32 / 2000.0))
        .collect();

    // pack codes per row
    let mut a_words = Vec::<u32>::new();
    for r in 0..m {
        a_words.extend(pack_row(&a_code[r * k..(r + 1) * k]));
    }
    let mut w_words = Vec::<u32>::new();
    for r in 0..n {
        w_words.extend(pack_row(&w_code[r * k..(r + 1) * k]));
    }

    // oracle: per unit, block = Σ centroid[a]·centroid[w]; acc += block·sa·sw
    let mut oracle = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f32;
            for u in 0..units {
                let mut block = 0.0f32;
                for j in 0..16 {
                    let c = u * 16 + j;
                    block += Q6F[a_code[mi * k + c] as usize] * Q6F[w_code[ni * k + c] as usize];
                }
                acc += block
                    * a_scale[mi * units + u].to_f32()
                    * w_scale[ni * units + u].to_f32();
            }
            oracle[mi * n + ni] = acc;
        }
    }

    let ah = client.create_from_slice(u32::as_bytes(&a_words));
    let ash = client.create_from_slice(f16::as_bytes(&a_scale));
    let wh = client.create_from_slice(u32::as_bytes(&w_words));
    let wsh = client.create_from_slice(f16::as_bytes(&w_scale));
    let outh = client.empty(m * n * 4);

    cubek_quant::qa_matmul::launch::<TestRuntime>(
        &client,
        ah,
        ash,
        wh,
        wsh,
        outh.clone(),
        m,
        n,
        k,
    );

    let bytes = client.read_one(outh).unwrap();
    let got = f32::from_bytes(&bytes);
    let cmax = oracle.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
    for (i, &g) in got.iter().enumerate() {
        let diff = (g - oracle[i]).abs();
        assert!(
            diff <= 1e-4 * (1.0 + cmax),
            "QA matmul [{i}]: expected {} got {g} (diff {diff})",
            oracle[i]
        );
    }
}
