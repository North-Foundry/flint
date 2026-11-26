/*!
 * Per-row evaluation for original and fused kernels.
 *
 * The evaluator is the compute kernel used by `run::execute`. It reads planar
 * `f32` samples from a bounded ring of input rows, interprets a stage's postfix
 * programs for each output pixel, and writes one planar output row. The same
 * interpreter is specialized for scalar `f32` and four-lane `f32x4`
 * arithmetic. Fused pointwise programs evaluate all live channel values per
 * pixel or SIMD group before writing the final planes.
 *
 * This module owns no image storage. `Scratch` provides reusable operand
 * stacks, while the caller guarantees that every sampled row is present in the
 * input ring and that the plan already validated offsets, channels, and stack
 * depth.
 */

use super::{buffer, offset, rows::Ring};
use crate::plan::{PointwiseProgram, Value};
use flint_sdk::{
    Expr, Result,
    graph::{Instruction, Stage},
};
use wide::f32x4;

#[cfg(test)]
mod tests;

/**
 * Reusable operands or temporary values for the selected backend.
 *
 * Each kernel uses the same scratch for either stack operands or fused
 * values. Every invocation writes a slot before reading it. The vector
 * allocation exists when a full SIMD group fits, alongside scalar scratch.
 */
pub(super) struct Scratch {
    scalar: Vec<f32>,
    vector: Vec<f32x4>,
}

impl Scratch {
    /**
     * Allocate workspace once for the maximum number of live slots.
     *
     * Allocate vector storage only when requested. Allocation failures become
     * `Error::Limit` through `buffer`.
     */
    pub(super) fn new(slots: usize, use_vectors: bool) -> Result<Self> {
        Ok(Self {
            scalar: buffer(slots, 0.0)?,
            vector: if use_vectors {
                buffer(slots, f32x4::ZERO)?
            } else {
                Vec::new()
            },
        })
    }

    /**
     * Return the allocated capacity in bytes for memory accounting.
     *
     * Capacities are measured instead of lengths because both stacks live for
     * the whole execution and contribute to the managed peak.
     */
    pub(super) fn capacity_bytes(&self) -> usize {
        self.scalar.capacity() * size_of::<f32>() + self.vector.capacity() * size_of::<f32x4>()
    }
}

/**
 * Execute a pointwise value program once per pixel or SIMD group, then write
 * its live output values to the corresponding planar channels.
 */
pub(super) fn evaluate_pointwise_row(
    program: &PointwiseProgram,
    input: &Ring,
    row_index: usize,
    output: &mut [f32],
    scratch: &mut Scratch,
) {
    let width = input.info().width() as usize;
    let row = input.row(row_index);
    let mut vector_end = 0;
    if !scratch.vector.is_empty() {
        vector_end = width / 4 * 4;
        for column in (0..vector_end).step_by(4) {
            evaluate_values(program, row, width, column, &mut scratch.vector);
            for (plane, &slot) in output.chunks_exact_mut(width).zip(&program.outputs) {
                plane[column..column + 4]
                    .copy_from_slice(&scratch.vector[usize::from(slot)].to_array());
            }
        }
    }
    for column in vector_end..width {
        evaluate_values(program, row, width, column, &mut scratch.scalar);
        for (plane, &slot) in output.chunks_exact_mut(width).zip(&program.outputs) {
            plane[column] = scratch.scalar[usize::from(slot)];
        }
    }
}

/**
 * Evaluate the live definitions in order, retaining final channels in scratch.
 * Lowering ensures each operand is initialized earlier in this invocation.
 */
#[inline]
fn evaluate_values<T: Lane>(
    program: &PointwiseProgram,
    row: &[f32],
    width: usize,
    column: usize,
    values: &mut [T],
) {
    for instruction in &program.instructions {
        let value = match instruction.value {
            Value::Input(channel) => {
                let start = usize::from(channel) * width;
                T::sample(&row[start..][..width], column, 0)
            }
            Value::Constant(bits) => T::splat(f32::from_bits(bits)),
            Value::Add(left, right) => values[usize::from(left)] + values[usize::from(right)],
            Value::Subtract(left, right) => values[usize::from(left)] - values[usize::from(right)],
            Value::Multiply(left, right) => values[usize::from(left)] * values[usize::from(right)],
            Value::Divide(left, right) => values[usize::from(left)] / values[usize::from(right)],
            Value::Minimum(left, right) => {
                values[usize::from(left)].min(values[usize::from(right)])
            }
            Value::Maximum(left, right) => {
                values[usize::from(left)].max(values[usize::from(right)])
            }
        };
        values[usize::from(instruction.destination)] = value;
    }
}

/**
 * Evaluate one output row of a stage into planar `f32` output.
 *
 * `input` must contain row `y` and every vertical neighbor the stage samples.
 * `output` is laid out as `channels × width`. A single-program stage applies
 * the same expression to every channel; otherwise program `i` computes channel
 * `i`. The SIMD path evaluates four adjacent pixels at a time, then the
 * remaining one to three pixels use the scalar path.
 *
 * This function has no error path because graph construction and planning have
 * already validated channel references, offsets, and depth.
 */
pub(super) fn evaluate_row(
    stage: &Stage,
    input: &Ring,
    y: usize,
    output: &mut [f32],
    scratch: &mut Scratch,
) {
    // Rows are planar: each channel occupies `width` contiguous samples.
    let width = input.info().width() as usize;
    let programs = stage.programs();
    for (channel, plane) in output.chunks_exact_mut(width).enumerate() {
        // A one-program stage broadcasts the same expression to all channels.
        let program = &programs[if programs.len() == 1 { 0 } else { channel }];
        let mut vector_end = 0;
        if !scratch.vector.is_empty() {
            // Four adjacent pixels share one f32x4 evaluation.
            let (groups, tail) = plane.as_chunks_mut::<4>();
            for (group, pixels) in groups.iter_mut().enumerate() {
                *pixels = evaluate(program, input, (group * 4, y), channel, &mut scratch.vector)
                    .to_array();
            }
            // Pixels after this index form the scalar tail.
            vector_end = width - tail.len();
        }
        // Handle the tail, or the whole row on the scalar backend.
        for (x, pixel) in plane.iter_mut().enumerate().skip(vector_end) {
            *pixel = evaluate(program, input, (x, y), channel, &mut scratch.scalar);
        }
    }
}

/**
 * Arithmetic specialization shared by the scalar and SIMD evaluators.
 *
 * A lane is one pixel for `f32` and four adjacent pixels for `f32x4`.
 * `sample` implements horizontal clamp-to-edge, and `min`/`max` normalize the
 * sign of zero so scalar and vector results agree.
 */
trait Lane:
    Copy
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
{
    /**
     * Broadcast one scalar value into every lane.
     */
    fn splat(value: f32) -> Self;

    /**
     * Read the sample at `x + dx`, clamping the horizontal offset to the row.
     */
    fn sample(row: &[f32], x: usize, dx: i32) -> Self;

    /**
     * Select the smaller lane value, treating negative zero as positive zero.
     */
    fn min(self, rhs: Self) -> Self;

    /**
     * Select the larger lane value, treating negative zero as positive zero.
     */
    fn max(self, rhs: Self) -> Self;
}

impl Lane for f32 {
    #[inline(always)]
    fn splat(value: f32) -> Self {
        // A scalar lane is the sample itself.
        value
    }
    #[inline(always)]
    fn sample(row: &[f32], x: usize, dx: i32) -> Self {
        // Reuse the runtime's clamp-to-edge offset helper.
        row[offset(x, dx, row.len())]
    }
    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        // Adding +0.0 normalizes -0.0 to +0.0.
        f32::min(self, rhs) + 0.0
    }
    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        // Adding +0.0 normalizes -0.0 to +0.0.
        f32::max(self, rhs) + 0.0
    }
}

impl Lane for f32x4 {
    #[inline(always)]
    fn splat(value: f32) -> Self {
        // Replicate the constant across all four lanes.
        f32x4::splat(value)
    }
    #[inline(always)]
    fn sample(row: &[f32], x: usize, dx: i32) -> Self {
        // Most pixels are interior: load four adjacent samples at once.
        if let Some(start) = x.checked_add_signed(dx as isize)
            && let Some(samples) = row.get(start..start + 4)
        {
            return f32x4::new(samples.try_into().expect("four contiguous samples"));
        }
        // At an edge, clamp each lane independently so a vector never reads
        // outside the row.
        f32x4::new(std::array::from_fn(|lane| {
            row[offset(x + lane, dx, row.len())]
        }))
    }
    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        // Match the scalar rule for the sign of zero.
        f32x4::min(self, rhs) + f32x4::ZERO
    }
    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        // Match the scalar rule for the sign of zero.
        f32x4::max(self, rhs) + f32x4::ZERO
    }
}

/**
 * Interpret one postfix program for a pixel or a group of four pixels.
 *
 * `(x, y)` is the first output coordinate and `channel` is the output channel
 * being computed. A sample without an explicit channel reads the active
 * channel; explicit channels are clamped to the last available channel, which
 * lets Gray inputs broadcast to all three components. `stack` is caller-owned
 * and must be at least as deep as the program's maximum live depth.
 */
#[inline]
fn evaluate<T: Lane>(
    program: &Expr,
    input: &Ring,
    (x, y): (usize, usize),
    channel: usize,
    stack: &mut [T],
) -> T {
    let info = input.info();
    let width = info.width() as usize;
    let height = info.height() as usize;
    // Clamp explicit channel references so RGB programs can read Gray inputs.
    let last_channel = info.format().channels() - 1;
    // Track the live operand count; the plan sized the stack for its maximum.
    let mut depth = 0;
    for &instruction in program.instructions() {
        match instruction {
            Instruction::Constant(value) => {
                // Push a broadcast constant.
                stack[depth] = T::splat(value);
                depth += 1;
            }
            Instruction::Sample {
                dx,
                dy,
                channel: selected,
            } => {
                // Vertical clamp: choose the nearest row at the top and bottom.
                let row = input.row(offset(y, dy, height));
                // Use the active channel unless the instruction selects one.
                let channel = selected.unwrap_or(channel).min(last_channel);
                // Select the channel plane, then sample horizontally.
                stack[depth] = T::sample(&row[channel * width..][..width], x, dx);
                depth += 1;
            }
            operator => {
                // Binary operator: pop two operands and push one result.
                let lhs = stack[depth - 2];
                let rhs = stack[depth - 1];
                depth -= 1;
                stack[depth - 1] = match operator {
                    Instruction::Add => lhs + rhs,
                    Instruction::Sub => lhs - rhs,
                    Instruction::Mul => lhs * rhs,
                    Instruction::Div => lhs / rhs,
                    Instruction::Min => lhs.min(rhs),
                    Instruction::Max => lhs.max(rhs),
                    _ => unreachable!("non-binary instruction"),
                };
            }
        }
    }
    // A validated program leaves exactly one result on the stack.
    stack[0]
}
