/*!
 * Managed working-memory accounting for execution plans.
 */

use super::{Backend, KernelPlan};
use flint_sdk::{Error, ImageInformation, Result};
use std::mem::size_of;
use wide::f32x4;

/**
 * Estimate the bytes needed by the runtime's reusable pixel and evaluator buffers.
 *
 * The estimate covers the packed source row, the packed output row, one
 * floating-point ring per kernel, the final output ring, and the shared stack
 * and value workspace. Codec state and immutable plan data are excluded.
 */
pub(super) fn estimate_managed_bytes(
    source: ImageInformation,
    output: ImageInformation,
    kernels: &[KernelPlan],
    backend: Backend,
) -> Result<usize> {
    let mut managed_bytes = source.row_bytes();
    if !kernels.is_empty() {
        managed_bytes = add(managed_bytes, output.row_bytes())?;
        for kernel in kernels {
            managed_bytes = add(
                managed_bytes,
                float_ring_bytes(kernel.input, kernel.buffered_rows)?,
            )?;
        }
        managed_bytes = add(managed_bytes, float_ring_bytes(output, 1)?)?;
        managed_bytes = add(
            managed_bytes,
            multiply(scratch_slots(kernels), lane_bytes(backend, source.width()))?,
        )?;
    }
    Ok(managed_bytes)
}

/**
 * Return the shared workspace length, since each kernel runs to completion
 * before another kernel uses the same scratch.
 */
pub(super) fn scratch_slots(kernels: &[KernelPlan]) -> usize {
    kernels
        .iter()
        .map(|kernel| kernel.stack_depth.max(kernel.temporary_values))
        .max()
        .unwrap_or(0)
}

/**
 * Return the scalar slot size plus vector storage when a full group fits.
 */
pub(super) fn lane_bytes(backend: Backend, width: u32) -> usize {
    size_of::<f32>()
        + if backend.uses_vectors(width) {
            size_of::<f32x4>()
        } else {
            0
        }
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
pub(crate) fn add(a: usize, b: usize) -> Result<usize> {
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
