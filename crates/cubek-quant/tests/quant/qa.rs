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
