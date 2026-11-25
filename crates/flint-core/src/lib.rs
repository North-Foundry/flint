//! Flint's row-streaming graph runtime.

mod plan;
mod registry;
mod run;

pub use plan::{Backend, ExecutionPlan, StagePlan};
pub use registry::{Registry, SelectedDecoder, SelectedEncoder};
pub use run::{ExecutionStatistics, execute};
