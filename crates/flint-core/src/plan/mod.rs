/*!
 * Execution planning for symbolic image graphs.
 *
 * Planning resolves the backend, lowers eligible pointwise stages to kernels,
 * and estimates working memory from the executable programs.
 */

mod backend;
mod execution_plan;
mod kernel_plan;
mod memory;
mod optimizer;
mod stage_plan;

pub use backend::Backend;
pub use execution_plan::ExecutionPlan;
pub use kernel_plan::KernelPlan;
pub use stage_plan::StagePlan;

pub(crate) use execution_plan::CompiledPlan;
pub(crate) use kernel_plan::KernelProgram;
pub(crate) use memory::{add, check_budget, float_ring_bytes};
pub(crate) use optimizer::{PointwiseProgram, Value};

/**
 * Select whether planning can fuse compatible pointwise stages.
 */
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Optimization {
    /** Select profitable fusion groups automatically. */
    #[default]
    Auto,
    /** Execute each original stage in its own kernel. */
    Disabled,
}
