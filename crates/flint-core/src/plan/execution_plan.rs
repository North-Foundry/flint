/*!
 * The resolved execution plan exposed by `explain`.
 */

use super::{
    backend::Backend,
    memory::estimate_managed_bytes,
    stage_plan::{StagePlan, analyze_stage, graph_stages},
};
use flint_sdk::{ImageExpression, ImageInformation, Result};
use std::fmt;

/**
 * The resolved backend, source and output metadata, and row requirements for a graph.
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
     * Stages in source-to-output order.
     */
    pub stages: Vec<StagePlan>,
    /**
     * Estimated bytes for the runtime's reusable pixel and evaluator buffers.
     */
    pub managed_bytes: usize,
}

impl ExecutionPlan {
    /**
     * Resolve the backend and estimate the working memory needed by the graph.
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
        let backend = backend.resolved();
        let mut stages: Vec<_> = graph_stages(image)
            .map(|(stage, output)| analyze_stage(stage, output))
            .collect();
        stages.reverse();
        let output = *image.metadata();
        let source = stages.first().map_or(output, |stage| stage.input);
        let managed_bytes = estimate_managed_bytes(source, output, &stages, backend)?;
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
