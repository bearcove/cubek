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
