/*!
 * Lower adjacent pointwise stages to a bounded, reusable value program.
 */

use flint_sdk::graph::{Instruction, Stage};
use std::collections::HashMap;

/**
 * Maximum channel-expanded source instructions examined for one fused program.
 * This also bounds the value graph, interning table, and lowering workspace.
 */
pub(super) const MAXIMUM_FUSION_INSTRUCTIONS: usize = 2048;

/**
 * Maximum simultaneously live values per scalar pixel or SIMD group.
 */
const MAXIMUM_TEMPORARY_VALUES: usize = 64;

/**
 * A pointwise kernel with one topologically ordered definition per live value.
 * An instruction reads its operands before replacing its destination slot.
 * Output slots stay live until the last instruction has executed.
 */
pub(crate) struct PointwiseProgram {
    pub(crate) instructions: Vec<ValueInstruction>,
    pub(crate) outputs: Vec<u8>,
    pub(crate) temporary_values: usize,
}

/**
 * Write one computed value to its reusable scratch slot.
 */
pub(crate) struct ValueInstruction {
    pub(crate) destination: u8,
    pub(crate) value: Value<u8>,
}

/**
 * A definition in either the planning value graph or the executable program.
 * Constants retain their exact floating-point bits, including signed zeros.
 */
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum Value<Reference> {
    Input(u8),
    Constant(u32),
    Add(Reference, Reference),
    Subtract(Reference, Reference),
    Multiply(Reference, Reference),
    Divide(Reference, Reference),
    Minimum(Reference, Reference),
    Maximum(Reference, Reference),
}

impl<Reference: Copy> Value<Reference> {
    /**
     * Translate operands in evaluation order while preserving the operator.
     */
    fn map<Mapped>(self, mut operand: impl FnMut(Reference) -> Mapped) -> Value<Mapped> {
        match self {
            Self::Input(channel) => Value::Input(channel),
            Self::Constant(bits) => Value::Constant(bits),
            Self::Add(left, right) => Value::Add(operand(left), operand(right)),
            Self::Subtract(left, right) => Value::Subtract(operand(left), operand(right)),
            Self::Multiply(left, right) => Value::Multiply(operand(left), operand(right)),
            Self::Divide(left, right) => Value::Divide(operand(left), operand(right)),
            Self::Minimum(left, right) => Value::Minimum(operand(left), operand(right)),
            Self::Maximum(left, right) => Value::Maximum(operand(left), operand(right)),
        }
    }
}

/**
 * Compose validated stages by resolving each sample to its preceding channel's
 * value. A broadcast program is first resolved for each active channel. Each
 * exact definition is interned with ordered operands, so no postfix program
 * is copied into its consumers.
 *
 * Returns `None` if the bounded candidate cannot be represented.
 */
pub(super) fn fuse(stages: &[(&Stage, super::StagePlan)]) -> Option<PointwiseProgram> {
    let input_channels = stages.first()?.1.input.format().channels();
    let mut definitions = Vec::new();
    let mut indices = HashMap::new();
    let mut operands = Vec::new();
    let mut channels = [0; 3];
    for (channel, input) in channels.iter_mut().take(input_channels).enumerate() {
        *input = intern(Value::Input(channel as u8), &mut definitions, &mut indices);
    }

    for (stage, analysis) in stages {
        let last_input = analysis.input.format().channels() - 1;
        let mut outputs = [0; 3];
        for (channel, output) in outputs
            .iter_mut()
            .take(analysis.output.format().channels())
            .enumerate()
        {
            let programs = stage.programs();
            let program = &programs[if programs.len() == 1 { 0 } else { channel }];
            for &instruction in program.instructions() {
                let value = match instruction {
                    Instruction::Sample {
                        dx: 0,
                        dy: 0,
                        channel: selected,
                    } => {
                        operands.push(channels[selected.unwrap_or(channel).min(last_input)]);
                        continue;
                    }
                    Instruction::Sample { .. } => return None,
                    Instruction::Constant(number) => Value::Constant(number.to_bits()),
                    operator => {
                        let right = operands.pop()?;
                        let left = operands.pop()?;
                        match operator {
                            Instruction::Add => Value::Add(left, right),
                            Instruction::Sub => Value::Subtract(left, right),
                            Instruction::Mul => Value::Multiply(left, right),
                            Instruction::Div => Value::Divide(left, right),
                            Instruction::Min => Value::Minimum(left, right),
                            Instruction::Max => Value::Maximum(left, right),
                            _ => unreachable!("non-binary instruction"),
                        }
                    }
                };
                operands.push(intern(value, &mut definitions, &mut indices));
            }
            *output = operands.pop()?;
            debug_assert!(operands.is_empty());
        }
        channels = outputs;
    }
    let output_channels = stages.last()?.1.output.format().channels();
    lower(&definitions, &channels[..output_channels])
}

/**
 * Return the index of an identical definition or append one after its operands.
 */
fn intern(
    value: Value<usize>,
    definitions: &mut Vec<Value<usize>>,
    indices: &mut HashMap<Value<usize>, usize>,
) -> usize {
    *indices.entry(value).or_insert_with(|| {
        let index = definitions.len();
        definitions.push(value);
        index
    })
}

/**
 * Count consumers reachable from the final channels, then assign slots to live
 * definitions in order. A producer's slot can be reused at its final consumer.
 * Every slot read therefore follows a write in the same evaluator invocation.
 */
fn lower(definitions: &[Value<usize>], outputs: &[usize]) -> Option<PointwiseProgram> {
    let mut uses = vec![0; definitions.len()];
    for &output in outputs {
        uses[output] += 1;
    }
    for (index, definition) in definitions.iter().enumerate().rev() {
        if uses[index] != 0 {
            definition.map(|operand| uses[operand] += 1);
        }
    }

    let mut slots = vec![0; definitions.len()];
    let mut available = Vec::new();
    let mut temporary_values = 0;
    let mut instructions = Vec::new();
    for (index, definition) in definitions.iter().enumerate() {
        if uses[index] == 0 {
            continue;
        }
        let value = definition.map(|operand| {
            uses[operand] -= 1;
            let slot = slots[operand];
            if uses[operand] == 0 {
                available.push(slot);
            }
            slot
        });
        let destination = if let Some(slot) = available.pop() {
            slot
        } else {
            if temporary_values == MAXIMUM_TEMPORARY_VALUES {
                return None;
            }
            let slot = temporary_values as u8;
            temporary_values += 1;
            slot
        };
        slots[index] = destination;
        instructions.push(ValueInstruction { destination, value });
    }

    Some(PointwiseProgram {
        instructions,
        outputs: outputs.iter().map(|&output| slots[output]).collect(),
        temporary_values,
    })
}

#[cfg(test)]
mod tests {
    use super::MAXIMUM_FUSION_INSTRUCTIONS;
    use crate::{Backend, ExecutionPlan};
    use flint_sdk::{Expr, ImageExpression, ImageInformation, Pixel, PixelFormat, Result};

    /**
     * Check that repeated channel references share the same producer and that
     * an unreachable input channel contributes no executable instruction.
     */
    #[test]
    fn repeated_inputs_share_a_live_definition() -> Result<()> {
        let image = ImageExpression::source(ImageInformation::new(128, 1, PixelFormat::Rgb8)?)
            .map_pixel(|pixel| Pixel::gray(pixel.r + pixel.g))?
            .map_color(|gray| gray.clone() * gray.clone() + gray)?;
        let plan = ExecutionPlan::analyze(&image, "test", Backend::Scalar)?;
        let [kernel] = plan.kernels.as_slice() else {
            panic!("expected one fused kernel");
        };
        assert_eq!(kernel.instruction_count, 5);
        assert_eq!(kernel.temporary_values, 2);
        Ok(())
    }

    /**
     * An allowed postfix expression can need more live values than a fused
     * kernel permits. Keep it standalone and continue fusing later stages.
     */
    #[test]
    fn excessive_live_values_leave_the_source_stage_standalone() -> Result<()> {
        let image = ImageExpression::source(ImageInformation::new(128, 1, PixelFormat::Gray8)?)
            .map_color(|value| {
                (1..=70).fold(value, |inner, constant| Expr::from(constant as f32) + inner)
            })?
            .map_color(|value| value * 0.125)?
            .map_color(|value| value - 5.0)?;
        let plan = ExecutionPlan::analyze(&image, "test", Backend::Scalar)?;
        let [original, fused] = plan.kernels.as_slice() else {
            panic!("expected one original kernel and one fused kernel");
        };
        assert_eq!(original.stages, 0..1);
        assert_eq!(original.temporary_values, 0);
        assert_eq!(fused.stages, 1..3);
        Ok(())
    }

    /**
     * Even a maximum-length valid graph is split at the source-program bound.
     */
    #[test]
    fn long_chains_respect_the_fusion_instruction_limit() -> Result<()> {
        let mut image = ImageExpression::source(ImageInformation::new(128, 1, PixelFormat::Rgb8)?);
        for index in 0..128 {
            image = image.map_color(|value| {
                ((value * 0.5 + index as f32).clamp(0.0, 1.0) - 0.2).clamp(0.0, 1.0)
            })?;
        }
        let plan = ExecutionPlan::analyze(&image, "test", Backend::Scalar)?;
        assert_eq!(plan.stages.len(), 128);
        assert!(plan.kernels.len() > 1);
        assert!(plan.kernels.len() < plan.stages.len());
        for kernel in &plan.kernels {
            let source_count: usize = plan.stages[kernel.stages.clone()]
                .iter()
                .map(|stage| stage.instruction_count)
                .sum();
            assert!(source_count <= MAXIMUM_FUSION_INSTRUCTIONS);
            assert!(kernel.instruction_count <= source_count);
        }
        Ok(())
    }
}
