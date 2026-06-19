use std::marker::PhantomData;

use crate::{
    args::MatmulArgs,
    components::batch::{
        BatchConfig as _, BatchMatmul, SliceIndex,
        base::BatchMatmulFamily,
        qa_gemv::{QaGemvBatchMatmulFamily, QaGemvBlueprint, QaGemvConfig},
    },
    definition::*,
};
use cubecl::{
    prelude::*,
    std::tensor::View,
    std::tensor::layout::Coords2d,
    {cube, num_traits::Zero},
};
use cubek_std::MatrixLayout;

#[cube(launch_unchecked, explicit_define, address_type = "dynamic")]
#[allow(clippy::type_complexity)]
/// Launches the quant GEMV kernel.
pub(crate) fn matmul_entry<
    Args: MatmulArgs<Config = ()>,
    Lhs: Numeric,
    LhsSize: Size,
    Rhs: Numeric,
    RhsSize: Size,
    Acc: Numeric,
    AccSize: Size,
>(
    inputs: &<Args as MatmulArgs>::Input<
        Vector<Lhs, LhsSize>,
        Vector<Rhs, RhsSize>,
        Vector<Acc, AccSize>,
    >,
    output: &mut <Args as MatmulArgs>::Output<Vector<Acc, AccSize>>,
    runtime_config: (),
    cube_mapping: CubeMapping,
    #[comptime] blueprint: QaGemvBlueprint,
    #[define(Lhs, Rhs, Acc)] _global: [StorageType; 3],
    #[define(LhsSize, RhsSize, AccSize)] _sizes: [usize; 3],
) {
    let state = Args::init_state::<Vector<Lhs, LhsSize>, Vector<Rhs, RhsSize>, Vector<Acc, AccSize>>(
        inputs,
        output,
        runtime_config,
        blueprint.lhs_global_layout_config(),
        blueprint.rhs_global_layout_config(),
        blueprint.out_global_layout_config(),
    );

    let vector_size_lhs = Args::view_lhs(&state).vector_size();
    let vector_size_rhs = Args::view_rhs(&state).vector_size();
    let vector_size_out = Args::view_out(&state).vector_size();
    let vector_sizes = comptime!(MatmulVectorSizes {
        lhs: vector_size_lhs,
        rhs: vector_size_rhs,
        out: vector_size_out,
    });

    let device_props = comptime::device_properties();
    let config = comptime!(QaGemvBatchMatmulFamily::expand_config(
        &device_props,
        &blueprint,
        &blueprint.dtypes,
        &vector_sizes
    ));

    if comptime!(config.is_err()) {
        push_validation_error(config.err().unwrap().to_string());
        comptime!(return);
    }
    let config = comptime!(config.unwrap());

    let state = Args::init_state::<Vector<Lhs, LhsSize>, Vector<Rhs, RhsSize>, Vector<Acc, AccSize>>(
        inputs,
        output,
        runtime_config,
        config.lhs_global_layout_config(),
        config.rhs_global_layout_config(),
        config.out_global_layout_config(),
    );

    let define!(RegisterLhs) = blueprint.dtypes.lhs_register;
    let define!(RegisterRhs) = blueprint.dtypes.rhs_register;
    let define!(RegisterAcc) = blueprint.dtypes.acc_register;

    QaGemvMatmul::<(
        (Lhs, LhsSize, Lhs, LhsSize, RegisterLhs, LhsSize),
        (Rhs, RhsSize, Rhs, RhsSize, RegisterRhs, RhsSize),
        (Acc, AccSize, Acc, AccSize, RegisterAcc, AccSize),
    )>::execute::<Args>(&state, cube_mapping, config);
}

pub struct QaGemvMatmul<MP: MatmulTypes> {
    _phantom: PhantomData<MP>,
}

#[cube]
impl<MT: MatmulTypes> BatchMatmul<(), MT> for QaGemvMatmul<MT> {
    type Config = QaGemvConfig;

    fn execute<Args: MatmulArgs>(
        state: &Args::State<LhsG<MT>, RhsG<MT>, AccG<MT>>,
        _cube_mapping: CubeMapping,
        #[comptime] _config: Self::Config,
    ) {
        // The decode GEMV: out[m, 0] = Σ_k lhs[m, k] · rhs[k, 0], with the WEIGHT
        // as lhs (the matrix, m = output channels) and the activation as rhs
        // (the vector, n = 1). One thread per output row m computes the full dot.
        // Reading lhs/rhs through the Args views and writing through view_out
        // makes it fusible — burn-fusion attaches the surrounding prologue/epilogue.
        let lhs = Args::view_lhs(state);
        let rhs = Args::view_rhs(state);
        let out = Args::view_out(state);

        let (_, _, k) = lhs.shape();
        let (_, size_m, _) = out.shape();

        // Flat thread → output row. The (x, y) grid axes index rows, z indexes the
        // batch (the cube_count is laid out accordingly in `qa_gemv_cube_count`).
        let row = ABSOLUTE_POS_X + ABSOLUTE_POS_Y * CUBE_COUNT_X * CUBE_DIM_X;
        let batch = ABSOLUTE_POS_Z as usize;

        let lhs_batch = Args::batch_lhs(state, batch);
        let lhs = lhs.view(SliceIndex::new(lhs_batch, lhs.shape()));
        let rhs_batch = Args::batch_rhs(state, batch);
        let rhs = rhs.view(SliceIndex::new(rhs_batch, rhs.shape()));
        let out_batch = Args::batch_out(state, batch);
        let mut out = out.view_mut(SliceIndex::new(out_batch, out.shape()));

        if row < size_m {
            let vector_size = comptime![Ord::max(lhs.vector_size(), rhs.vector_size())];
            let size!(NA) = vector_size;
            let mut acc = AccRE::<MT>::zero();
            for kb in range_stepped(0u32, k, vector_size as u32) {
                let lhs_vec = load_unrolled::<_, _, NA>(&lhs, (row, kb), MatrixLayout::RowMajor);
                let rhs_vec = load_unrolled::<_, _, NA>(&rhs, (kb, 0), MatrixLayout::ColMajor);
                let prod = Vector::<AccRE<MT>, NA>::cast_from(lhs_vec)
                    * Vector::<AccRE<MT>, NA>::cast_from(rhs_vec);
                #[unroll]
                for v in 0..vector_size {
                    acc += prod.extract(v);
                }
            }
            out.write((row, 0), Vector::cast_from(acc));
        }
    }
}

#[cube]
fn load_unrolled<I: Numeric, N: Size, N2: Size>(
    view: &View<Vector<I, N>, Coords2d>,
    pos: Coords2d,
    #[comptime] layout: MatrixLayout,
) -> Vector<I, N2> {
    let vector_size = N2::value();
    comptime![assert!(vector_size >= view.vector_size())];
    let view_vector_size = view.vector_size();
    if comptime![view.vector_size() == vector_size] {
        Vector::cast_from(view.read(pos))
    } else {
        let (row, col) = pos;
        let mut out = Vector::empty();
        #[unroll]
        for i in range_stepped(0, vector_size as u32, view_vector_size as u32) {
            let pos = match layout {
                MatrixLayout::RowMajor => (row, col + i),
                MatrixLayout::ColMajor => (row + i, col),
            };
            let value = view.read(pos);
            #[unroll]
            for n in 0..view_vector_size {
                out.insert(i as usize + n, value.extract(n));
            }
        }
        out
    }
}
