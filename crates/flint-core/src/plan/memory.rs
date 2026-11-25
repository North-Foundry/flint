/*!
 * Managed working-memory accounting for execution plans.
 */

use super::{backend::Backend, stage_plan::StagePlan};
use flint_sdk::{Error, ImageInformation, Result};
use std::mem::size_of;
use wide::f32x4;

/**
 * Estimate the bytes needed by the runtime's reusable pixel and evaluator buffers.
 *
 * The estimate covers the packed source row, the packed output row, one
 * floating-point ring per stage, the final output ring, and the evaluator
 * operand stacks. Codec-owned state and graph metadata are excluded.
 */
pub(super) fn estimate_managed_bytes(
    source: ImageInformation,
    output: ImageInformation,
    stages: &[StagePlan],
    backend: Backend,
) -> Result<usize> {
    let mut managed_bytes = source.row_bytes();
    let mut max_stack = 0;
    if !stages.is_empty() {
        managed_bytes = add(managed_bytes, output.row_bytes())?;
        for stage in stages {
            managed_bytes = add(
                managed_bytes,
                float_ring_bytes(stage.input, stage.buffered_rows)?,
            )?;
            max_stack = max_stack.max(stage.stack_depth);
        }
        managed_bytes = add(managed_bytes, float_ring_bytes(output, 1)?)?;
        let scratch = size_of::<f32>()
            + if backend == Backend::Simd {
                size_of::<f32x4>()
            } else {
                0
            };
        managed_bytes = add(managed_bytes, multiply(max_stack, scratch)?)?;
    }
    Ok(managed_bytes)
}

/**
 * Reject a plan whose estimated or measured working memory exceeds the budget.
 */
pub(crate) fn check_budget(bytes: usize, limit: usize) -> Result<()> {
    if bytes > limit {
        return Err(Error::Limit(format!(
            "pipeline needs {bytes} managed bytes; budget is {limit}"
        )));
    }
    Ok(())
}

/**
 * Calculate the byte capacity for the requested number of floating-point rows.
 */
pub(crate) fn float_ring_bytes(info: ImageInformation, rows: usize) -> Result<usize> {
    multiply(multiply(info.row_bytes(), rows)?, size_of::<f32>())
}

/**
 * Add two sizes, reporting overflow as a resource limit.
 */
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or_else(size_error)
}

/**
 * Multiply two sizes, reporting overflow or an unaddressable result as a limit.
 */
fn multiply(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(size_error)
}

/**
 * Build the shared error for working-buffer size overflow.
 */
fn size_error() -> Error {
    Error::Limit("working buffer size overflow".into())
}
