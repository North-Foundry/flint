use flint_sdk::{
    Error, ImageExpression, ImageInformation, Result,
    graph::{Instruction, Node, Primitive, Stage},
};
use std::{fmt, mem::size_of};
use wide::f32x4;

/// Scalar reference execution, automatic CPU selection, or SIMD with scalar fallback.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Backend {
    #[default]
    Auto,
    Scalar,
    Simd,
}

impl Backend {
    pub fn resolved(self) -> Self {
        if self != Self::Scalar && simd_available() {
            Self::Simd
        } else {
            Self::Scalar
        }
    }
}

fn simd_available() -> bool {
    // wide uses explicit SIMD when these baseline features are compiled in.
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
    ))]
    {
        std::arch::is_x86_feature_detected!("sse2")
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        std::arch::is_aarch64_feature_detected!("neon")
    }
    #[cfg(not(any(
        all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "sse2"
        ),
        all(target_arch = "aarch64", target_feature = "neon"),
    )))]
    {
        false
    }
}

/**
 * An operation and the relative rectangle it needs from its input.
 */
#[derive(Clone, Debug)]
pub struct StagePlan {
    pub primitive: Primitive,
    pub input: ImageInformation,
    pub output: ImageInformation,
    pub min_x: i32,
    pub max_x: i32,
    pub min_y: i32,
    pub max_y: i32,
    pub buffered_rows: usize,
    pub(crate) stack_depth: usize,
}

/**
 * The resolved backend, source and output metadata, and row requirements for a graph.
 */
#[derive(Clone, Debug)]
pub struct ExecutionPlan {
    pub source_format: &'static str,
    pub source: ImageInformation,
    pub output: ImageInformation,
    pub backend: Backend,
    pub stages: Vec<StagePlan>,
    /// Estimated bytes for the runtime's reusable pixel and evaluator buffers.
    pub managed_bytes: usize,
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

impl ExecutionPlan {
    /**
     * Resolve the backend and estimate the working memory needed by the graph.
     */
    pub fn analyze(
        image: &ImageExpression,
        source_format: &'static str,
        backend: Backend,
    ) -> Result<Self> {
        let backend = backend.resolved();
        let mut stages: Vec<_> = graph_stages(image)
            .map(|(stage, output)| analyze_stage(stage, output))
            .collect();
        stages.reverse();
        let output = *image.metadata();
        let source = stages.first().map_or(output, |stage| stage.input);
        let mut managed_bytes = source.row_bytes();
        let mut max_stack = 0;
        if !stages.is_empty() {
            managed_bytes = add(managed_bytes, output.row_bytes())?;
            for stage in &stages {
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
        Ok(Self {
            source_format,
            source,
            output,
            backend,
            stages,
            managed_bytes,
        })
    }
}

pub(crate) fn check_budget(bytes: usize, limit: usize) -> Result<()> {
    if bytes > limit {
        return Err(Error::Limit(format!(
            "pipeline needs {bytes} managed bytes; budget is {limit}"
        )));
    }
    Ok(())
}

/**
 * Derive the stage's spatial dependencies and evaluator stack size from its programs.
 */
fn analyze_stage(stage: &Stage, output: ImageInformation) -> StagePlan {
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
                    depth += 1;
                }
                Instruction::Constant(_) => depth += 1,
                _ => depth -= 1,
            }
            stack_depth = stack_depth.max(depth);
        }
    }
    if min_x == i32::MAX {
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
        stack_depth,
    }
}

/**
 * Calculate the byte capacity for the requested number of floating-point rows.
 */
pub(crate) fn float_ring_bytes(info: ImageInformation, rows: usize) -> Result<usize> {
    multiply(multiply(info.row_bytes(), rows)?, size_of::<f32>())
}

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or_else(size_error)
}

fn multiply(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(size_error)
}

fn size_error() -> Error {
    Error::Limit("working buffer size overflow".into())
}

impl fmt::Display for ExecutionPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {}x{} {:?} → {:?}; {:?} backend; estimated managed working memory: {} B",
            self.source_format,
            self.source.width(),
            self.source.height(),
            self.source.format(),
            self.output.format(),
            self.backend,
            self.managed_bytes
        )?;
        for (index, stage) in self.stages.iter().enumerate() {
            write!(
                f,
                "\n  {}. {:?}: {:?} → {:?}, x={}..={}, y={}..={}, {} buffered input row(s)",
                index + 1,
                stage.primitive,
                stage.input.format(),
                stage.output.format(),
                stage.min_x,
                stage.max_x,
                stage.min_y,
                stage.max_y,
                stage.buffered_rows
            )?;
        }
        Ok(())
    }
}
