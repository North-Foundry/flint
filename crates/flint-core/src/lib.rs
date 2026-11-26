//! Flint's row-streaming graph runtime.

mod codec;
mod plan;
mod run;

pub use codec::{Registry, SelectedDecoder, SelectedEncoder};
pub use plan::{Backend, ExecutionPlan, KernelPlan, Optimization, StagePlan};
pub use run::{ExecutionOptions, ExecutionStatistics, execute, execute_with_options};
