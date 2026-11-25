//! Open an image lazily, compose operations, and save with automatic codec selection.

mod engine;
mod image;

pub use engine::{Engine, EngineBuilder};
pub use flint_core::{Backend, ExecutionPlan, ExecutionStatistics, StagePlan};
pub use flint_sdk as sdk;
pub use flint_sdk::{Error, Result};
pub use image::Image;
