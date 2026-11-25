/*!
 * Execution planning for symbolic image graphs.
 *
 * Planning resolves the backend, derives each stage's spatial requirements and
 * evaluator stack depth, and estimates the managed working memory needed to run
 * the graph. The submodules separate backend detection, stage analysis, memory
 * accounting, and the public execution plan.
 */

mod backend;
mod execution_plan;
mod memory;
mod stage_plan;

pub use backend::Backend;
pub use execution_plan::ExecutionPlan;
pub use stage_plan::StagePlan;

pub(crate) use memory::{check_budget, float_ring_bytes};
pub(crate) use stage_plan::graph_stages;
