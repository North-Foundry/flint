/*!
 * Executable kernel programs and their public planning metadata.
 */

use super::{
    Backend, Optimization, StagePlan,
    memory::{float_ring_bytes, lane_bytes},
    optimizer::{MAXIMUM_FUSION_INSTRUCTIONS, PointwiseProgram, fuse},
};
use flint_sdk::{ImageInformation, graph::Stage};
use std::ops::Range;

/**
 * A single streaming kernel and the original stages it represents.
 */
#[derive(Clone, Debug)]
pub struct KernelPlan {
    /**
     * Zero-based range into `ExecutionPlan::stages`, in source-to-output order.
     */
    pub stages: Range<usize>,
    /** Metadata of the input rows. */
    pub input: ImageInformation,
    /** Metadata of the output rows. */
    pub output: ImageInformation,
    /** Leftmost horizontal sample offset. */
    pub min_x: i32,
    /** Rightmost horizontal sample offset. */
    pub max_x: i32,
    /** Topmost vertical sample offset. */
    pub min_y: i32,
    /** Bottommost vertical sample offset. */
    pub max_y: i32,
    /** Number of input rows the kernel may need at once. */
    pub buffered_rows: usize,
    /** Instructions executed for all output channels of one pixel or SIMD group. */
    pub instruction_count: usize,
    /** Reusable value slots needed by a fused program. */
    pub temporary_values: usize,
    /** Reusable operand slots needed by a standalone postfix program. */
    pub(crate) stack_depth: usize,
}

/**
 * An immutable program referencing either an original stage or lowered values.
 */
pub(crate) enum KernelProgram<'graph> {
    Original(&'graph Stage),
    Pointwise(PointwiseProgram),
}

/**
 * Select bounded pointwise runs and leave all other stages as standalone kernels.
 * When a candidate fails, test successively shorter prefixes before proceeding.
 */
pub(super) fn select_kernels<'graph>(
    stages: &[(&'graph Stage, StagePlan)],
    optimization: Optimization,
    backend: Backend,
) -> (Vec<KernelPlan>, Vec<KernelProgram<'graph>>) {
    let mut kernels = Vec::new();
    let mut start = 0;
    while let Some((stage, analysis)) = stages.get(start) {
        if optimization == Optimization::Auto && analysis.is_pointwise() {
            let mut instruction_count = 0;
            let mut length = stages[start..]
                .iter()
                .take_while(|(_, stage)| {
                    instruction_count += stage.instruction_count;
                    stage.is_pointwise() && instruction_count <= MAXIMUM_FUSION_INSTRUCTIONS
                })
                .count();
            while length >= 2 {
                let end = start + length;
                let group = &stages[start..end];
                if let Some(program) =
                    fuse(group).filter(|program| profitable(group, program, backend))
                {
                    let mut kernel = original_kernel(start, analysis);
                    kernel.stages.end = end;
                    kernel.output = group[length - 1].1.output;
                    kernel.instruction_count = program.instructions.len();
                    kernel.temporary_values = program.temporary_values;
                    kernel.stack_depth = 0;
                    kernels.push((kernel, KernelProgram::Pointwise(program)));
                    start = end;
                    break;
                }
                length = length.div_ceil(2).min(length - 1);
            }
            if length >= 2 {
                continue;
            }
        }
        kernels.push((
            original_kernel(start, analysis),
            KernelProgram::Original(stage),
        ));
        start += 1;
    }
    kernels.into_iter().unzip()
}

/**
 * Use fusion only when value dispatch does not grow and the eliminated rows
 * pay for any extra scratch capacity. Both calculations are per candidate.
 */
fn profitable(group: &[(&Stage, StagePlan)], program: &PointwiseProgram, backend: Backend) -> bool {
    let original_instructions: usize = group.iter().map(|(_, stage)| stage.instruction_count).sum();
    let original_stack = group
        .iter()
        .map(|(_, stage)| stage.stack_depth)
        .max()
        .unwrap_or(0);
    let extra_slots = program.temporary_values.saturating_sub(original_stack);
    // Both the value cap and the number of bytes per lane are small constants.
    let extra_bytes = extra_slots * lane_bytes(backend, group[0].1.input.width());
    let eliminated_bytes = group[1..].iter().try_fold(0usize, |total, (_, stage)| {
        total.checked_add(float_ring_bytes(stage.input, 1).ok()?)
    });
    program.instructions.len() <= original_instructions
        && eliminated_bytes.is_some_and(|bytes| bytes > extra_bytes)
}

/**
 * Use the original stage's analysis unchanged for a standalone kernel.
 */
fn original_kernel(original_index: usize, stage: &StagePlan) -> KernelPlan {
    KernelPlan {
        stages: original_index..original_index + 1,
        input: stage.input,
        output: stage.output,
        min_x: stage.min_x,
        max_x: stage.max_x,
        min_y: stage.min_y,
        max_y: stage.max_y,
        buffered_rows: stage.buffered_rows,
        instruction_count: stage.instruction_count,
        temporary_values: 0,
        stack_depth: stage.stack_depth,
    }
}
