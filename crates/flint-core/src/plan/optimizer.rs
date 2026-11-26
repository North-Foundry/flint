/*!
 * Fold constant subexpressions and lower adjacent pointwise stages to a bounded value program.
 */

use flint_sdk::graph::{Instruction, Stage};
use std::collections::HashMap;

use crate::run::fold_constants;

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
pub(super) fn fuse(
    stages: &[(&Stage, super::StagePlan)],
    use_vectors: bool,
) -> Option<(PointwiseProgram, usize)> {
    let input_channels = stages.first()?.1.input.format().channels();
    let mut definitions = Vec::new();
    let mut indices = HashMap::new();
    let mut operands = Vec::new();
    let mut folded_instructions = 0;
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
                        if let (Value::Constant(left), Value::Constant(right)) =
                            (definitions[left], definitions[right])
                            && let Some(result) = fold_constants(
                                operator,
                                f32::from_bits(left),
                                f32::from_bits(right),
                                use_vectors,
                            )
                        {
                            folded_instructions += 1;
                            Value::Constant(result.to_bits())
                        } else {
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
    Some((
        lower(&definitions, &channels[..output_channels])?,
        folded_instructions,
    ))
}

/**
 * Compact a validated postfix program in one pass. Two literal instructions
 * immediately preceding a binary operator are its complete operand trees.
 * Replacing that suffix can make its parent eligible. Leave samples and their
 * order intact, and allocate only once a safe replacement has been found.
 */
pub(super) fn fold_postfix(
    source: &[Instruction],
    use_vectors: bool,
) -> Option<(Vec<Instruction>, usize)> {
    let (start, first_result) = source.windows(3).enumerate().find_map(|(start, window)| {
        let &[
            Instruction::Constant(left),
            Instruction::Constant(right),
            operator,
        ] = window
        else {
            return None;
        };
        fold_constants(operator, left, right, use_vectors).map(|result| (start, result))
    })?;
    let mut instructions = Vec::with_capacity((start + 1).max(4));
    instructions.extend_from_slice(&source[..start]);
    instructions.push(Instruction::Constant(first_result));
    let mut folded_instructions = 1;
    for &operator in &source[start + 3..] {
        if let [
            ..,
            Instruction::Constant(left),
            Instruction::Constant(right),
        ] = instructions.as_slice()
            && let Some(result) = fold_constants(operator, *left, *right, use_vectors)
        {
            instructions.truncate(instructions.len() - 2);
            instructions.push(Instruction::Constant(result));
            folded_instructions += 1;
        } else {
            instructions.push(operator);
        }
    }
    Some((instructions, folded_instructions))
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
    use super::{MAXIMUM_FUSION_INSTRUCTIONS, fold_postfix};
    use crate::{Backend, ExecutionPlan};
    use flint_sdk::{
        Error, Expr, ImageExpression, ImageInformation, Pixel, PixelFormat, Result,
        graph::Instruction,
    };

    /**
     * Fold nested constant operands in expression order, leaving an unsafe
     * operation and its consumer in the executable postfix sequence.
     */
    #[test]
    fn postfix_folds_nested_arithmetic_and_keeps_nan_producers() {
        let source = (Expr::from(2.0) + 3.0) * 4.0 + (Expr::from(0.0) / 0.0).max(0.25);
        let (instructions, count) = fold_postfix(source.instructions(), true).unwrap();
        assert_eq!(count, 2);
        assert!(matches!(
            instructions.as_slice(),
            [
                Instruction::Constant(20.0),
                Instruction::Constant(0.0),
                Instruction::Constant(0.0),
                Instruction::Div,
                Instruction::Constant(0.25),
                Instruction::Max,
                Instruction::Add,
            ]
        ));
        assert_eq!(source.instructions().len(), 11);
    }

    /**
     * Sample-dependent stencil arithmetic retains the original evaluator path.
     */
    #[test]
    fn stencil_without_folds_uses_the_original_program() -> Result<()> {
        let image = ImageExpression::source(ImageInformation::new(7, 1, PixelFormat::Gray8)?)
            .stencil(|neighbors| neighbors.at(-1, 0) * 0.5)?;
        let plan = crate::plan::CompiledPlan::prepare(
            &image,
            "test",
            Backend::Scalar,
            crate::Optimization::Auto,
            usize::MAX,
        )?;
        assert!(matches!(
            plan.programs.as_slice(),
            [crate::plan::KernelProgram::Original(_)]
        ));
        Ok(())
    }

    /**
     * Folded constants share exact values with literals and other folded
     * subexpressions. Lower only reachable definitions and reuse their slots.
     */
    #[test]
    fn folded_definitions_share_values_and_allow_dead_definition_removal() -> Result<()> {
        let image = ImageExpression::source(ImageInformation::new(128, 1, PixelFormat::Gray8)?)
            .map_pixel(|pixel| {
                Pixel::rgb(
                    pixel.r + (Expr::from(3.0) * 2.0),
                    Expr::from(7.0) + 1.0,
                    Expr::from(2.0) + 4.0,
                )
            })?
            .map_pixel(|pixel| {
                Pixel::rgb(pixel.r.clone(), pixel.r + (pixel.b.clone() - 6.0), pixel.b)
            })?;
        let plan = ExecutionPlan::analyze(&image, "test", Backend::Scalar)?;
        let [kernel] = plan.kernels.as_slice() else {
            panic!("expected a fused kernel");
        };
        assert_eq!(kernel.folded_instructions, 4);
        // Input, 6, input + 6, 0, (input + 6) + 0. Keep the input-dependent add.
        assert_eq!(kernel.instruction_count, 5);
        assert_eq!(kernel.temporary_values, 3);
        Ok(())
    }

    /**
     * Apply the fusion cap to source instructions even when all of a stage's
     * operations can be folded to a single executable instruction.
     */
    #[test]
    fn folding_does_not_bypass_the_fusion_source_limit() -> Result<()> {
        let terms = MAXIMUM_FUSION_INSTRUCTIONS / 2;
        let image = ImageExpression::source(ImageInformation::new(128, 1, PixelFormat::Gray8)?)
            .map_color(|_| (0..terms).fold(Expr::from(0.0), |total, _| total + 1.0))?
            .map_color(|value| value / 4096.0)?;
        let plan = ExecutionPlan::analyze(&image, "test", Backend::Scalar)?;
        assert!(plan.stages[0].instruction_count > MAXIMUM_FUSION_INSTRUCTIONS);
        assert_eq!(plan.kernels.len(), 2);
        assert_eq!(plan.kernels[0].folded_instructions, terms);
        assert_eq!(plan.kernels[0].instruction_count, 1);
        assert_eq!(plan.kernels[1].instruction_count, 3);
        Ok(())
    }

    /**
     * Size fusion candidates after folding, even when the original postfix
     * operand stack would exceed the live value limit for a fused program.
     */
    #[test]
    fn folding_can_make_a_candidate_fit_the_live_value_limit() -> Result<()> {
        let image = ImageExpression::source(ImageInformation::new(128, 1, PixelFormat::Gray8)?)
            .map_color(|sample| {
                let coefficient = (1..=70).fold(Expr::from(0.0), |inner, term| term as f32 + inner);
                sample / coefficient
            })?
            .map_color(|value| value * 80.0)?;
        let plan = ExecutionPlan::analyze(&image, "test", Backend::Scalar)?;
        assert!(plan.stages[0].stack_depth > super::MAXIMUM_TEMPORARY_VALUES);
        let [kernel] = plan.kernels.as_slice() else {
            panic!("expected a fused kernel");
        };
        assert_eq!(kernel.folded_instructions, 70);
        assert_eq!(kernel.instruction_count, 5);
        assert_eq!(kernel.temporary_values, 2);
        Ok(())
    }

    /**
     * Compare a borderline fusion candidate against folded postfix scratch.
     * Separate only the group that costs more than its standalone kernels.
     */
    #[test]
    fn fusion_accounts_for_smaller_folded_standalone_stacks() -> Result<()> {
        let image = ImageExpression::source(ImageInformation::new(1, 1, PixelFormat::Gray8)?)
            .map_color(|value| value + (0.125 + (0.125 + Expr::from(0.125))))?
            .map_pixel(|pixel| Pixel::rgb(pixel.r + 0.0625, pixel.g + 0.125, pixel.b + 0.25))?
            .stencil(|neighbors| neighbors.at(-1, 0))?
            .map_pixel(|pixel| Pixel::gray(pixel.r))?
            .map_color(|value| value)?
            .map_color(|value| value)?;
        let plan = ExecutionPlan::analyze_with_optimization(
            &image,
            "test",
            Backend::Scalar,
            crate::Optimization::Auto,
            46,
        )?;
        let ranges: Vec<_> = plan
            .kernels
            .iter()
            .map(|kernel| kernel.stages.clone())
            .collect();
        assert_eq!(ranges, [0..1, 1..2, 2..3, 3..6]);
        assert_eq!(plan.kernels[0].folded_instructions, 2);
        assert_eq!(plan.managed_bytes, 46);
        Ok(())
    }

    /**
     * Keep SDK construction limits even when the graph would collapse to one constant.
     */
    #[test]
    fn constant_only_graphs_are_subject_to_sdk_limits() -> Result<()> {
        let source = ImageExpression::source(ImageInformation::new(1, 1, PixelFormat::Gray8)?);
        let mut expression = Expr::from(1.0);
        for _ in 0..16 {
            expression = expression.clone() + expression;
        }
        assert!(matches!(
            source.clone().map_color(|_| expression),
            Err(Error::Limit(_))
        ));

        let mut image = source;
        for _ in 0..128 {
            image = image.map_color(|_| Expr::from(0.25) + 0.25)?;
        }
        assert!(matches!(
            image.map_color(|_| Expr::from(0.5)),
            Err(Error::Limit(_))
        ));
        Ok(())
    }

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
