use cubek_std::MatrixLayout;

use crate::components::{batch::BatchConfig, global::memory::GlobalLayoutConfig};

/// Config for the single-pass warp-per-row quant GEMV. Same operand layouts as
/// the naive matmul: lhs (the weight matrix) row-major `[m, k]`, rhs (the
/// activation vector) col-major `[k, n]`, out row-major `[m, n]`.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct QaGemvConfig {}

impl BatchConfig for QaGemvConfig {
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
}
