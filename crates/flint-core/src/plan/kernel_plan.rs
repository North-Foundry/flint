/*!
 * Executable kernel programs and their public planning metadata.
 */

use super::{
    Backend, Optimization, StagePlan,
    memory::{float_ring_bytes, lane_bytes},
    optimizer::{MAXIMUM_FUSION_INSTRUCTIONS, PointwiseProgram, fold_postfix, fuse},
    stage_plan::analyze_programs,
};
use flint_sdk::{
    ImageInformation,
    graph::{Instruction, Stage},
};
use std::{borrow::Cow, ops::Range};

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
    /**
     * Source arithmetic occurrences successfully folded during planning, counted
     * per output channel before deduplication and dead definition removal.
     */
    pub folded_instructions: usize,
    /** Reusable value slots needed by a fused program. */
    pub temporary_values: usize,
    /** Reusable operand slots needed by a standalone postfix program. */
    pub(crate) stack_depth: usize,
}

/**
 * An immutable program for a standalone stage or a fused pointwise kernel.
 * Borrow an unmodified stage. For a folded stage, own only the changed
 * instructions and borrow the other channel programs.
 */
pub(crate) enum KernelProgram<'graph> {
    Original(&'graph Stage),
    FoldedPostfix(Vec<Cow<'graph, [Instruction]>>),
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
        let use_vectors = backend.uses_vectors(analysis.input.width());
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
                if let Some((program, folded_instructions)) = fuse(group, use_vectors)
                    .filter(|(program, folded)| profitable(group, program, *folded, backend))
                {
                    let mut kernel = original_kernel(start, analysis);
                    kernel.stages.end = end;
                    kernel.output = group[length - 1].1.output;
                    kernel.instruction_count = program.instructions.len();
                    kernel.folded_instructions = folded_instructions;
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
        kernels.push(standalone_kernel(
            start,
            stage,
            analysis,
            optimization,
            backend,
        ));
        start += 1;
    }
    kernels.into_iter().unzip()
}

/**
 * Keep the standalone evaluator and rebuild its analysis only if a program
 * changed. A broadcast program is folded once and counted for every channel.
 */
pub(super) fn standalone_kernel<'graph>(
    index: usize,
    stage: &'graph Stage,
    analysis: &StagePlan,
    optimization: Optimization,
    backend: Backend,
) -> (KernelPlan, KernelProgram<'graph>) {
    let programs = stage.programs();
    if optimization == Optimization::Auto {
        let use_vectors = backend.uses_vectors(analysis.input.width());
        let mut folded = Vec::new();
        let mut folded_instructions = 0;
        for (index, program) in programs.iter().enumerate() {
            let source = program.instructions();
            if let Some((instructions, count)) = fold_postfix(source, use_vectors) {
                if folded.is_empty() {
                    folded.extend(
                        programs[..index]
                            .iter()
                            .map(|program| Cow::Borrowed(program.instructions())),
                    );
                }
                folded.push(Cow::Owned(instructions));
                folded_instructions += count;
            } else if !folded.is_empty() {
                folded.push(Cow::Borrowed(source));
            }
        }
        if folded_instructions != 0 {
            let executable =
                analyze_programs(stage, analysis.output, folded.iter().map(Cow::as_ref));
            let mut kernel = original_kernel(index, &executable);
            kernel.folded_instructions = if programs.len() == 1 {
                folded_instructions * analysis.output.format().channels()
            } else {
                folded_instructions
            };
            return (kernel, KernelProgram::FoldedPostfix(folded));
        }
    }
    (
        original_kernel(index, analysis),
        KernelProgram::Original(stage),
    )
}

/**
 * Use fusion only when value dispatch does not grow and the eliminated rows
 * pay for any extra scratch capacity. Both calculations are per candidate.
 */
fn profitable(
    group: &[(&Stage, StagePlan)],
    program: &PointwiseProgram,
    folded_instructions: usize,
    backend: Backend,
) -> bool {
    let original_instructions: usize = group.iter().map(|(_, stage)| stage.instruction_count).sum();
    if program.instructions.len() > original_instructions {
        return false;
    }
    let original_stack = group
        .iter()
        .map(|(_, stage)| stage.stack_depth)
        .max()
        .unwrap_or(0);
    let Some(eliminated_bytes) = group[1..].iter().try_fold(0usize, |total, (_, stage)| {
        total.checked_add(float_ring_bytes(stage.input, 1).ok()?)
    }) else {
        return false;
    };
    let slot_bytes = lane_bytes(backend, group[0].1.input.width());
    let saves_memory = |stack| {
        let extra_slots = program.temporary_values.saturating_sub(stack);
        eliminated_bytes > extra_slots * slot_bytes
    };
    if !saves_memory(original_stack) {
        return false;
    }
    // Every postfix program needs at least one slot. Only optimize the
    // standalone alternative if a smaller stack could change this decision.
    if folded_instructions == 0 || saves_memory(1) {
        return true;
    }
    let folded_stack = group
        .iter()
        .enumerate()
        .map(|(index, (stage, analysis))| {
            standalone_kernel(index, stage, analysis, Optimization::Auto, backend)
                .0
                .stack_depth
        })
        .max()
        .unwrap_or(0);
    saves_memory(folded_stack)
}

/**
 * Convert the analysis of a standalone postfix stage into kernel metadata.
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
        folded_instructions: 0,
        temporary_values: 0,
        stack_depth: stage.stack_depth,
    }
}
