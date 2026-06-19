use cubecl::{
    CubeCount, CubeDim, Runtime,
    client::ComputeClient,
    ir::{AddressType, DeviceProperties},
    server::LaunchError,
};
use cubek_std::MatrixLayout;

use crate::{
    args::*,
    components::{
        CubeDimResource,
        batch::{
            BatchMatmulFamily,
            qa_gemv::{QaGemvConfig, QaGemvMatmul, matmul_entry},
        },
        global::memory::GlobalLayoutConfig,
        stage::NumStages,
    },
    definition::{
        Blueprint, CubeMappingLaunch, MatmulElems, MatmulProblem, MatmulSetupError, MatmulTypes,
        MatmulVectorSizes, SwizzleModes, TilingScheme,
    },
};

/// Number of output rows (planes) handled per cube.
pub(crate) const QA_GEMV_NUM_PLANES: u32 = 8;

/// Single-pass warp-per-row GEMV family for the quant decode MatVec (n = 1).
pub struct QaGemvBatchMatmulFamily {}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct QaGemvBlueprint {
    pub dtypes: MatmulElems,
}

impl Blueprint for QaGemvBlueprint {
    fn lhs_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: MatrixLayout::RowMajor,
            check_row_bounds: false,
            check_col_bounds: false,
        }
    }

    fn rhs_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: MatrixLayout::ColMajor,
            check_row_bounds: false,
            check_col_bounds: false,
        }
    }

    fn out_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: MatrixLayout::RowMajor,
            check_row_bounds: false,
            check_col_bounds: false,
        }
    }

    fn tiling_scheme(&self) -> TilingScheme {
        panic!("QaGemv Blueprint doesn't have a TilingScheme")
    }

    fn swizzle_modes(&self) -> SwizzleModes {
        panic!("QaGemv Blueprint doesn't have Swizzle Modes")
    }
}

impl BatchMatmulFamily<()> for QaGemvBatchMatmulFamily {
    type Matmul<MP: MatmulTypes> = QaGemvMatmul<MP>;
    type Config = QaGemvConfig;
    type Blueprint = QaGemvBlueprint;

    fn expand_config(
        _device_props: &DeviceProperties,
        _blueprint: &Self::Blueprint,
        _dtypes: &MatmulElems,
        _vector_sizes: &MatmulVectorSizes,
    ) -> Result<Self::Config, MatmulSetupError> {
        Ok(QaGemvConfig {})
    }

    fn num_stages() -> NumStages {
        (1, 1).into()
    }

    unsafe fn launch_unchecked<MA: MatmulArgs<Config = ()>, R: Runtime>(
        client: &ComputeClient<R>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        address_type: AddressType,
        input: InputRuntimeArg<MA, R>,
        output: OutputRuntimeArg<MA, R>,
        _config: ConfigRuntimeArg<MA, R>,
        cube_mapping: CubeMappingLaunch<R>,
        blueprint: QaGemvBlueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), LaunchError> {
        unsafe {
            matmul_entry::launch_unchecked::<MA, Lhs, LhsSize, Rhs, RhsSize, Acc, AccSize, R>(
                client,
                cube_count,
                cube_dim,
                address_type,
                input,
                output,
                (),
                cube_mapping,
                blueprint,
                [dtypes.lhs_global, dtypes.rhs_global, dtypes.acc_global],
                [vector_sizes.lhs, vector_sizes.rhs, vector_sizes.out],
            )
        };

        Ok(())
    }

    fn cubedim_resource(
        _blueprint: &Self::Blueprint,
        _dtypes: &MatmulElems,
        _vector_sizes: &MatmulVectorSizes,
    ) -> Result<CubeDimResource, MatmulSetupError> {
        Ok(CubeDimResource::Planes(QA_GEMV_NUM_PLANES))
    }

    fn validate_blueprint<R: Runtime>(
        _client: &ComputeClient<R>,
        _blueprint: &Self::Blueprint,
        problem: &MatmulProblem,
        _dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        // The kernel writes one scalar per output row.
        if vector_sizes.out > 1 {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "QaGemv: vector size on output not supported",
            )));
        }
        // Lanes step over K in units of the (vectorized) load width; K must be a
        // whole number of those, with no remainder, since the kernel doesn't
        // bound-check the K loop.
        let load = vector_sizes.lhs.max(vector_sizes.rhs);
        if !problem.k.is_multiple_of(load) {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "QaGemv: k={} must be divisible by the load width {}",
                problem.k, load,
            ))));
        }
        Ok(())
    }
}
