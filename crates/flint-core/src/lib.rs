//! Flint's row-streaming graph runtime.

mod codec;
mod plan;
mod run;

pub use codec::{Registry, SelectedDecoder, SelectedEncoder};
pub use plan::{Backend, ExecutionPlan, StagePlan};
pub use run::{ExecutionStatistics, execute};
