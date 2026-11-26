/*!
 * The resolved execution plan exposed by `explain`.
 */

use super::{
    Backend, KernelPlan, Optimization,
    kernel_plan::{KernelProgram, select_kernels},
    memory::{estimate_managed_bytes, lane_bytes, scratch_slots},
    stage_plan::{StagePlan, analyze_stage, graph_stages},
};
use flint_sdk::{ImageExpression, ImageInformation, Result};
use std::fmt;

/**
 * The resolved backend, original stages, and executable row kernels for a graph.
 */
#[derive(Clone, Debug)]
pub struct ExecutionPlan {
    /**
     * Name of the decoder selected for the source.
     */
    pub source_format: &'static str,
    /**
     * Metadata of the rows supplied by the source.
     */
    pub source: ImageInformation,
    /**
     * Metadata of the rows produced by the graph.
     */
    pub output: ImageInformation,
    /**
     * Backend selected after resolving automatic detection.
     */
    pub backend: Backend,
    /**
     * Optimization mode requested for this plan.
     */
    pub optimization: Optimization,
    /**
     * Stages in source-to-output order.
     */
    pub stages: Vec<StagePlan>,
    /**
     * Executable kernels in source-to-output order.
     */
    pub kernels: Vec<KernelPlan>,
    /**
     * Estimated bytes for the runtime's reusable pixel and evaluator buffers.
     */
    pub managed_bytes: usize,
}

impl ExecutionPlan {
    /**
     * Plan with automatic optimization and without a configured memory budget.
     *
     * # Errors
     *
     * Returns `Error::Limit` when working-buffer size arithmetic overflows.
     */
    pub fn analyze(
        image: &ImageExpression,
        source_format: &'static str,
        backend: Backend,
    ) -> Result<Self> {
        Self::analyze_with_optimization(
            image,
            source_format,
            backend,
            Optimization::Auto,
            usize::MAX,
        )
    }

    /**
     * Choose kernels using the requested optimization mode and memory budget.
     * An over-budget graph still reports its estimated requirement if it can
     * be represented. Execution checks that estimate before allocating buffers.
     *
     * # Errors
     *
     * Returns `Error::Limit` when working-buffer size arithmetic overflows.
     */
    pub fn analyze_with_optimization(
        image: &ImageExpression,
        source_format: &'static str,
        backend: Backend,
        optimization: Optimization,
        memory_limit: usize,
    ) -> Result<Self> {
        Ok(
            CompiledPlan::prepare(image, source_format, backend, optimization, memory_limit)?
                .report,
        )
    }

    /**
     * Return the number of original intermediate row buffers removed by fusion.
     */
    pub fn eliminated_buffers(&self) -> usize {
        self.stages.len().saturating_sub(self.kernels.len())
    }

    /**
     * Return the capacity in bytes required by the largest temporary-value
     * workspace. The allocation is shared with standalone operand stacks.
     */
    pub fn temporary_workspace_bytes(&self) -> usize {
        self.kernels
            .iter()
            .map(|kernel| kernel.temporary_values)
            .max()
            .unwrap_or(0)
            * lane_bytes(self.backend, self.source.width())
    }

    /**
     * Return the capacity common to temporary values and operand stacks.
     */
    pub(crate) fn scratch_slots(&self) -> usize {
        scratch_slots(&self.kernels)
    }
}

/**
 * The public report and the matching executable programs for a single run.
 */
pub(crate) struct CompiledPlan<'graph> {
    pub(crate) report: ExecutionPlan,
    pub(crate) programs: Vec<KernelProgram<'graph>>,
}

impl<'graph> CompiledPlan<'graph> {
    /**
     * Analyze the source graph once, then choose and size the actual kernels.
     * If the optimized plan exceeds the budget, consider the unfused plan.
     */
    pub(crate) fn prepare(
        image: &'graph ImageExpression,
        source_format: &'static str,
        backend: Backend,
        optimization: Optimization,
        memory_limit: usize,
    ) -> Result<Self> {
        let backend = backend.resolved();
        let mut stages: Vec<_> = graph_stages(image)
            .map(|(stage, output)| (stage, analyze_stage(stage, output)))
            .collect();
        stages.reverse();
        let output = *image.metadata();
        let source = stages.first().map_or(output, |(_, stage)| stage.input);
        let (mut kernels, mut programs) = select_kernels(&stages, optimization, backend);
        let mut managed_bytes = estimate_managed_bytes(source, output, &kernels, backend);
        if kernels.len() < stages.len()
            && !managed_bytes
                .as_ref()
                .is_ok_and(|&bytes| bytes <= memory_limit)
        {
            let (original_kernels, original_programs) =
                select_kernels(&stages, Optimization::Disabled, backend);
            let original_bytes = estimate_managed_bytes(source, output, &original_kernels, backend);
            if let Ok(bytes) = original_bytes
                && !managed_bytes
                    .as_ref()
                    .is_ok_and(|&optimized| optimized <= bytes)
            {
                kernels = original_kernels;
                programs = original_programs;
                managed_bytes = Ok(bytes);
            }
        }
        let report = ExecutionPlan {
            source_format,
            source,
            output,
            backend,
            optimization,
            stages: stages.into_iter().map(|(_, stage)| stage).collect(),
            kernels,
            managed_bytes: managed_bytes?,
        };
        Ok(Self { report, programs })
    }
}

impl fmt::Display for ExecutionPlan {
    /**
     * Summarize the executable kernels, their provenance, and estimated storage.
     */
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
        write!(
            f,
            "\n  {:?} optimization, {} original stage(s), {} executable kernel(s), {} intermediate buffer(s) eliminated, {} B temporary workspace",
            self.optimization,
            self.stages.len(),
            self.kernels.len(),
            self.eliminated_buffers(),
            self.temporary_workspace_bytes()
        )?;
        for (index, kernel) in self.kernels.iter().enumerate() {
            write!(f, "\n  {}. ", index + 1,)?;
            let original = self.stages.get(kernel.stages.clone());
            if let Some([stage]) = original {
                write!(f, "stage {} {:?}", kernel.stages.start + 1, stage.primitive)?;
            } else {
                write!(
                    f,
                    "fused stages {}–{}",
                    kernel.stages.start + 1,
                    kernel.stages.end
                )?;
            }
            write!(
                f,
                ": {:?} → {:?}, x={}..={}, y={}..={}, {} buffered input row(s), {} instruction(s), {} temporary value(s)",
                kernel.input.format(),
                kernel.output.format(),
                kernel.min_x,
                kernel.max_x,
                kernel.min_y,
                kernel.max_y,
                kernel.buffered_rows,
                kernel.instruction_count,
                kernel.temporary_values
            )?;
            if let Some(stages) = original
                && stages.len() > 1
            {
                let original_count: usize =
                    stages.iter().map(|stage| stage.instruction_count).sum();
                write!(f, " (from {original_count} original instructions)")?;
            }
        }
        Ok(())
    }
}
