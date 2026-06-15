use cubecl::prelude::*;
use cubecl::{
    Runtime, ir::ElemType, ir::FloatKind, server::CopyDescriptor, std::tensor::TensorHandle,
    {TestRuntime, zspace::shape},
};
use cubek_quant::{
    scheme::QuantLevel, scheme::QuantMode, scheme::QuantParam, scheme::QuantScheme,
    scheme::QuantStore, scheme::QuantValue,
};

// Outermost TQ4 centroid and the largest half-gap between adjacent centroids
// (at the tails) — the worst-case nearest-centroid reconstruction error / scale.
const Q4F_MAX_CENTROID: f32 = 2.732590;
const Q4F_MAX_HALF_GAP: f32 = (2.732590 - 2.069017) / 2.0;

/// TQ4 (table codebook): values map to nearest Lloyd-Max centroid; error is the
/// centroid spacing.
#[test]
fn test_codebook_q4f_roundtrip() {
    // map data into the centroid range [-2.73, 2.73]; error = max half-gap * scale.
    roundtrip(QuantValue::Q4F, Q4F_MAX_CENTROID, Q4F_MAX_HALF_GAP);
}

/// TQ8 (linear codebook = affine offset-binary): value = (code - 128) * scale;
/// error is half a quant step (scale/2).
#[test]
fn test_codebook_q8f_roundtrip() {
    // map data into [-127, 127] (code range, offset-binary); error = scale/2.
    roundtrip(QuantValue::Q8F, 127.0, 0.5);
}

// 64-level Lloyd-Max TQ6 centroids (matches cubek-quant's table).
const Q6F_CENTROIDS: [f32; 64] = [
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

/// TQ6 dense dequant: 6-bit codes pack densely (code j at bit 6j), straddling
/// u32 words — bee's exact layout. Build a packed input by hand, dequantize on
/// GPU, and check it equals `centroid[code] * scale` bit-for-bit (mod f32).
#[test]
fn test_codebook_q6f_dense_dequant() {
    let (m, n) = (16usize, 64usize); // n divisible by 16 (the dense unit)
    let scale_f32 = 0.37f32;
    let words_per_row = n * 6 / 32; // 12 u32 per row
    let client = TestRuntime::client(&Default::default());

    // deterministic pseudo-random 6-bit codes, and the dense little-endian pack.
    let mut codes = vec![0u8; m * n];
    let mut s = 0x1234_5678_9abc_def0u64;
    for c in codes.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *c = (s % 64) as u8;
    }
    let mut words = vec![0u32; m * words_per_row];
    for row in 0..m {
        let wbase = row * words_per_row;
        for j in 0..n {
            let code = (codes[row * n + j] & 0x3f) as u32;
            let gb = j * 6;
            let (word, bitoff) = (gb / 32, gb % 32);
            words[wbase + word] |= code << bitoff;
            if bitoff + 6 > 32 {
                words[wbase + word + 1] |= code >> (32 - bitoff);
            }
        }
    }

    let in_alloc = client.create_tensor_from_slice(
        u32::as_bytes(&words),
        shape![m, words_per_row],
        u32::type_size(),
    );
    let scale_alloc =
        client.create_tensor_from_slice(f32::as_bytes(&[scale_f32]), shape![1], f32::type_size());
    let input = TensorHandle::new(
        in_alloc.memory,
        shape![m, words_per_row],
        in_alloc.strides,
        u32::as_type_native_unchecked(),
    );
    let scale = TensorHandle::new(
        scale_alloc.memory,
        shape![1],
        scale_alloc.strides,
        f32::as_type_native_unchecked(),
    );
    let output_f = TensorHandle::zeros(&client, shape![m, n], f32::as_type_native_unchecked());

    let scheme = QuantScheme::default()
        .with_level(QuantLevel::Tensor)
        .with_mode(QuantMode::Codebook)
        .with_value(QuantValue::Q6F)
        .with_store(QuantStore::PackedU32Dense(0))
        .with_param(QuantParam::F32);

    cubek_quant::dequantize::launch_ref(
        &client,
        input.binding(),
        output_f.clone().binding(),
        scale.binding(),
        &scheme,
        f32::as_type_native_unchecked().storage_type(),
    )
    .unwrap();

    let computed = client.read_one_unchecked_tensor(CopyDescriptor::new(
        output_f.handle.clone().binding(),
        output_f.shape().clone(),
        output_f.strides().clone(),
        core::mem::size_of::<f32>(),
    ));
    let got = f32::from_bytes(&computed);

    assert_eq!(got.len(), m * n);
    for (i, &g) in got.iter().enumerate() {
        let expected = Q6F_CENTROIDS[codes[i] as usize] * scale_f32;
        let diff = (g - expected).abs();
        assert!(
            diff <= 1e-4 * (1.0 + expected.abs()),
            "TQ6 dense [{i}] code {}: expected {expected} got {g} (diff {diff})",
            codes[i]
        );
    }
}

/// Quantize → dequantize round-trip: choose a scale that maps the data into the
/// representable range, then assert reconstruction within `err_per_scale * scale`.
fn roundtrip(value: QuantValue, range_divisor: f32, err_per_scale: f32) {
    let (m, n) = (16usize, 64usize);
    let client = TestRuntime::client(&Default::default());
    let shape = shape![m, n];

    let num_elems = m * n;
    let half = num_elems as f32 / 2.0;
    let data: Vec<f32> = (0..num_elems).map(|v| v as f32 - half).collect();

    let max_abs = data.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
    let scale_f32 = max_abs / range_divisor;
    let data_scale = vec![scale_f32];

    let input_alloc =
        client.create_tensor_from_slice(f32::as_bytes(&data), shape.clone(), f32::type_size());
    let scale_alloc =
        client.create_tensor_from_slice(f32::as_bytes(&data_scale), shape![1], f32::type_size());

    let input = TensorHandle::new(
        input_alloc.memory,
        shape.clone(),
        input_alloc.strides,
        f32::as_type_native_unchecked(),
    );
    let scale = TensorHandle::new(
        scale_alloc.memory,
        shape![1],
        scale_alloc.strides,
        f32::as_type_native_unchecked(),
    );
    let output_f = TensorHandle::zeros(&client, shape, f32::as_type_native_unchecked());

    let scheme = QuantScheme::default()
        .with_level(QuantLevel::Tensor)
        .with_mode(QuantMode::Codebook)
        .with_value(value)
        .with_store(QuantStore::PackedU32(0))
        .with_param(QuantParam::F32);

    // Output shape is in packed u32s.
    let shape_out = shape![m, n / scheme.num_quants()];
    let [output_alloc, output_scale_alloc] = client
        .empty_tensors(vec![
            cubecl::server::MemoryLayoutDescriptor {
                strategy: cubecl::server::MemoryLayoutStrategy::Contiguous,
                shape: shape_out.clone(),
                elem_size: u32::type_size(),
            },
            cubecl::server::MemoryLayoutDescriptor {
                strategy: cubecl::server::MemoryLayoutStrategy::Contiguous,
                shape: shape![1],
                elem_size: f32::type_size(),
            },
        ])
        .try_into()
        .unwrap();
    let output = TensorHandle::new(
        output_alloc.memory,
        shape_out,
        output_alloc.strides,
        u32::as_type_native_unchecked(),
    );
    let output_scale = TensorHandle::new(
        output_scale_alloc.memory,
        shape![1],
        output_scale_alloc.strides,
        f32::as_type_native_unchecked(),
    );

    cubek_quant::quantize::launch_ref(
        &client,
        input.binding(),
        output.clone().binding(),
        scale.binding(),
        output_scale.clone().binding(),
        &scheme,
        ElemType::Float(FloatKind::Flex32),
    )
    .unwrap();

    cubek_quant::dequantize::launch_ref(
        &client,
        output.binding(),
        output_f.clone().binding(),
        output_scale.clone().binding(),
        &scheme,
        f32::as_type_native_unchecked().storage_type(),
    )
    .unwrap();

    let computed = client.read_one_unchecked_tensor(CopyDescriptor::new(
        output_f.handle.clone().binding(),
        output_f.shape().clone(),
        output_f.strides().clone(),
        core::mem::size_of::<f32>(),
    ));
    let restored = f32::from_bytes(&computed);

    let max_error = err_per_scale * scale_f32 * (1.0 + 1e-4);
    assert_eq!(restored.len(), data.len());
    for (actual, expected) in restored.iter().zip(&data) {
        let diff = (actual - expected).abs();
        assert!(
            diff <= max_error,
            "codebook roundtrip: expected {expected} got {actual} (diff {diff} > {max_error})"
        );
    }
}
