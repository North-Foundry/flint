/*!
 * Stage dependency analysis and graph traversal.
 */

use flint_sdk::{
    ImageExpression, ImageInformation,
    graph::{Instruction, Node, Primitive, Stage},
};

/**
 * An operation and the relative rectangle it needs from its input.
 */
#[derive(Clone, Debug)]
pub struct StagePlan {
    /**
     * Kind of operation performed by the stage.
     */
    pub primitive: Primitive,
    /**
     * Metadata of the rows supplied to the stage.
     */
    pub input: ImageInformation,
    /**
     * Metadata of the rows produced by the stage.
     */
    pub output: ImageInformation,
    /**
     * Leftmost horizontal offset sampled by the stage.
     */
    pub min_x: i32,
    /**
     * Rightmost horizontal offset sampled by the stage.
     */
    pub max_x: i32,
    /**
     * Topmost vertical offset sampled by the stage.
     */
    pub min_y: i32,
    /**
     * Bottommost vertical offset sampled by the stage.
     */
    pub max_y: i32,
    /**
     * Number of input rows the stage may need at once.
     */
    pub buffered_rows: usize,
    /**
     * Instructions executed for all output channels of one pixel or SIMD group.
     */
    pub instruction_count: usize,
    /**
     * Maximum live operands in the stage's postfix programs.
     */
    pub(crate) stack_depth: usize,
}

/**
 * Walk from the output toward the source, yielding each stage and its output metadata.
 */
pub(crate) fn graph_stages(
    image: &ImageExpression,
) -> impl Iterator<Item = (&Stage, ImageInformation)> {
    let mut current = image;
    std::iter::from_fn(move || match current.node() {
        Node::Source => None,
        Node::Stage(stage) => {
            let output = *current.metadata();
            current = stage.input();
            Some((stage, output))
        }
    })
}

/**
 * Derive the stage's spatial dependencies and evaluator stack size from its programs.
 */
pub(super) fn analyze_stage(stage: &Stage, output: ImageInformation) -> StagePlan {
    let mut min_x = i32::MAX;
    let mut max_x = i32::MIN;
    let mut min_y = i32::MAX;
    let mut max_y = i32::MIN;
    let mut stack_depth = 0;
    for program in stage.programs() {
        let mut depth = 0;
        for instruction in program.instructions() {
            match *instruction {
                Instruction::Sample { dx, dy, .. } => {
                    min_x = min_x.min(dx);
                    max_x = max_x.max(dx);
                    min_y = min_y.min(dy);
                    max_y = max_y.max(dy);
                    // A sample pushes one operand.
                    depth += 1;
                }
                Instruction::Constant(_) => {
                    // A constant pushes one operand.
                    depth += 1;
                }
                _ => {
                    // Every binary operator pops two operands and pushes one.
                    depth -= 1;
                }
            }
            stack_depth = stack_depth.max(depth);
        }
    }
    if min_x == i32::MAX {
        // A stage without spatial samples reads only the current pixel.
        (min_x, max_x, min_y, max_y) = (0, 0, 0, 0);
    }
    let buffered_rows = ((max_y - min_y + 1) as usize).min(output.height() as usize);
    StagePlan {
        primitive: stage.primitive(),
        input: *stage.input().metadata(),
        output,
        min_x,
        max_x,
        min_y,
        max_y,
        buffered_rows,
        instruction_count: if stage.programs().len() == 1 {
            stage.programs()[0].instructions().len() * output.format().channels()
        } else {
            stage
                .programs()
                .iter()
                .map(|program| program.instructions().len())
                .sum()
        },
        stack_depth,
    }
}

impl StagePlan {
    /**
     * Report whether all symbolic sample offsets are zero.
     */
    pub(super) fn is_pointwise(&self) -> bool {
        (self.min_x, self.max_x, self.min_y, self.max_y) == (0, 0, 0, 0)
    }
}
