pub mod launch;

use cubecl::{CubeCount, CubeDim, Runtime, client::ComputeClient, ir::AddressType};
use cubek_std::cube_count::CubeCountPlan;

use crate::{
    args::{ConfigRuntimeArg, InputRuntimeArg, MatmulArgs, OutputRuntimeArg},
    components::{
        batch::{
            BatchMatmulFamily,
            qa_gemv::{QaGemvBatchMatmulFamily, QaGemvBlueprint, QA_GEMV_NUM_PLANES},
        },
        stage::NumStages,
    },
    definition::{
        CubeMappingLaunch, MatmulAvailabilityError, MatmulElems, MatmulProblem, MatmulSetupError,
        MatmulVectorSizes,
    },
    routines::{
        BatchMatmulRoutine, BlueprintStrategy, DeviceSettings, ExpandInfo, LaunchInfo, Routine,
        batch_validate_blueprint,
    },
};

/// Single-pass warp-per-row quant GEMV routine — a fusible MatVec candidate the
/// autotuner can select. One plane per output row, lanes reduce K via plane_sum.
pub struct QaGemvRoutine {}

#[derive(Default, Clone)]
pub struct QaGemvStrategy {}

impl std::fmt::Display for QaGemvStrategy {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl From<()> for QaGemvStrategy {
    fn from(_value: ()) -> Self {
        Self {}
    }
}

impl Routine<()> for QaGemvRoutine {
    type Strategy = QaGemvStrategy;
    type Blueprint = QaGemvBlueprint;
}

impl BatchMatmulRoutine<()> for QaGemvRoutine {
    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    fn launch<MA: MatmulArgs<Config = ()>, R: Runtime>(
        client: &ComputeClient<R>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        address_type: AddressType,
        input: InputRuntimeArg<MA, R>,
        output: OutputRuntimeArg<MA, R>,
        _config: ConfigRuntimeArg<MA, R>,
        cube_count_input: CubeMappingLaunch<R>,
        blueprint: Self::Blueprint,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        unsafe {
            <QaGemvBatchMatmulFamily>::launch_unchecked::<MA, R>(
                client,
                cube_dim,
                cube_count,
                address_type,
                input,
                output,
                (),
                cube_count_input,
                blueprint,
                dtypes,
                vector_sizes,
            )?
        }
        Ok(())
    }

    #[allow(clippy::result_large_err)]
    fn validate_blueprint<R: Runtime>(
        client: &ComputeClient<R>,
        blueprint: &Self::Blueprint,
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
        vector_sizes: &MatmulVectorSizes,
    ) -> Result<(), MatmulSetupError> {
        batch_validate_blueprint::<QaGemvBatchMatmulFamily, (), R>(
            client,
            blueprint,
            problem,
            dtypes,
            vector_sizes,
        )
    }

    fn num_stages() -> NumStages {
        QaGemvBatchMatmulFamily::num_stages()
    }

    fn expand_blueprint<R: cubecl::Runtime>(
        problem: &MatmulProblem,
        _device_settings: &DeviceSettings<R>,
        _strategy: &BlueprintStrategy<(), Self>,
    ) -> Result<ExpandInfo<Self::Blueprint>, MatmulSetupError> {
        let dtypes = MatmulElems::from_globals(&problem.global_dtypes);
        let blueprint = QaGemvBlueprint {
            dtypes: dtypes.clone(),
        };
        Ok(ExpandInfo { blueprint, dtypes })
    }

    fn prepare<R: cubecl::Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        expand_info: ExpandInfo<Self::Blueprint>,
    ) -> Result<LaunchInfo<Self::Blueprint>, MatmulSetupError> {
        let ExpandInfo { blueprint, dtypes } = expand_info;

        Self::validate_blueprint(
            &device_settings.client,
            &blueprint,
            problem,
            &dtypes,
            &device_settings.vector_sizes,
        )?;

        let cube_dim = QaGemvBatchMatmulFamily::cubedim_resource(
            &blueprint,
            &dtypes,
            &device_settings.vector_sizes,
        )?
        .to_cube_dim(device_settings.plane_dim)?;

        Ok(LaunchInfo {
            blueprint,
            dtypes,
            cube_dim,
            cube_count_plan: qa_gemv_cube_count(&problem.out_shape, cube_dim.x * cube_dim.y)?,
            address_type: problem.address_type,
            vector_sizes: device_settings.vector_sizes,
        })
    }
}

/// One thread per output row: the kernel flattens the (x, y) grid axes into the
/// row index, so we need `ceil(m / threads_per_cube)` cubes along Y (X stays a
/// single cube). `z` indexes the batch.
#[allow(clippy::result_large_err)]
fn qa_gemv_cube_count(
    output_shape: &[usize],
    threads_per_cube: u32,
) -> Result<CubeCountPlan, MatmulSetupError> {
    let ndims = output_shape.len();
    let m = output_shape[ndims - 2];

    let row_cubes = f32::ceil(m as f32 / threads_per_cube as f32) as u32;
    let mut batch_cubes = 1u32;
    #[allow(clippy::needless_range_loop)]
    for i in 0..ndims - 2 {
        batch_cubes *= output_shape[i] as u32;
    }

    let cube_count_plan = CubeCountPlan::new_from_problem((1, row_cubes.max(1), batch_cubes).into());
    let max_cube_count = u16::MAX as u32;
    if row_cubes > max_cube_count || batch_cubes > max_cube_count {
        return Err(MatmulSetupError::Unavailable(
            MatmulAvailabilityError::CubeCountTooBig(cube_count_plan.resolve()),
        ));
    }

    let _ = QA_GEMV_NUM_PLANES;
    Ok(cube_count_plan)
}
