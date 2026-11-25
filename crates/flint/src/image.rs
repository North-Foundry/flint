use crate::{Engine, ExecutionPlan, ExecutionStatistics};
use flint_core::{SelectedDecoder, SelectedEncoder, execute};
use flint_sdk::{Destination, Error, ImageExpression, ImageInformation, Operation, Result, Source};
use std::{
    path::Path,
    sync::{Arc, OnceLock},
};

/**
 * A reusable symbolic image whose operations run when the image is saved or written.
 *
 * Clones share the source and the existing expression graph.
 */
#[derive(Clone)]
pub struct Image {
    engine: Engine,
    expression: ImageExpression,
    decoder: SelectedDecoder,
    source: Arc<dyn Fn() -> Result<Source> + Send + Sync>,
}

impl Image {
    /**
     * Create the source node from inspected metadata and retain the stream factory
     * for future executions.
     */
    pub(super) fn from_source(
        engine: Engine,
        metadata: ImageInformation,
        decoder: SelectedDecoder,
        source: impl Fn() -> Result<Source> + Send + Sync + 'static,
    ) -> Self {
        Self {
            engine,
            expression: ImageExpression::source(metadata),
            decoder,
            source: Arc::new(source),
        }
    }

    /**
     * Inspect an image file using the shared default engine.
     *
     * Each execution reopens the file from the beginning.
     */
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        static DEFAULT: OnceLock<Engine> = OnceLock::new();
        DEFAULT
            .get_or_init(|| Engine::builder().build().expect("valid default engine"))
            .open(path)
    }

    /**
     * Return the output dimensions and pixel format described by the current graph.
     */
    pub fn metadata(&self) -> &ImageInformation {
        self.expression.metadata()
    }

    /**
     * Consume this image and use the operation to construct the returned image's graph.
     * Pixel processing begins when the image is saved or written.
     *
     * # Errors
     *
     * Propagates errors from the operation and returns [`Error::Invalid`] if the
     * resulting graph has a different source.
     */
    pub fn apply(mut self, operation: impl Operation) -> Result<Self> {
        let expression = operation.apply(self.expression.clone())?;
        if !self.expression.shares_source(&expression) {
            return Err(Error::Invalid(
                "operation returned a graph with a different source".into(),
            ));
        }
        self.expression = expression;
        Ok(self)
    }

    /**
     * Analyze the expression graph and metadata to report the selected backend,
     * row dependencies, and estimated working memory.
     */
    pub fn explain(&self) -> Result<ExecutionPlan> {
        ExecutionPlan::analyze(
            &self.expression,
            self.decoder.name(),
            self.engine.settings().backend,
        )
    }

    /**
     * Save using the encoder for the destination's file extension.
     *
     * The destination is replaced only after the staged output completes successfully.
     */
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.save_with_statistics(path).map(drop)
    }

    /**
     * Save the image as in [`Self::save`] and report the selected backend and peak
     * capacity of the managed working buffers.
     */
    pub fn save_with_statistics(&self, path: impl AsRef<Path>) -> Result<ExecutionStatistics> {
        let path = std::path::absolute(path)?;
        let extension = path
            .extension()
            .and_then(|extension| extension.to_str())
            .ok_or_else(|| {
                Error::Unsupported("destination must have a recognized extension".into())
            })?;
        let encoder = self.engine.settings().registry.encoder(extension)?;
        let parent = path
            .parent()
            .filter(|parent_path| !parent_path.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let temporary = tempfile::NamedTempFile::new_in(parent)?;
        let statistics = self.write_to(encoder, Box::new(temporary.reopen()?))?;
        temporary.persist(path).map_err(|error| error.error)?;
        Ok(statistics)
    }

    /**
     * Write to a sequential destination using the encoder for the given extension.
     *
     * An execution error may leave partial output in the destination.
     */
    pub fn write(&self, extension: &str, destination: Destination) -> Result<()> {
        self.write_with_statistics(extension, destination).map(drop)
    }

    /**
     * Write the image as in [`Self::write`] and report the selected backend and peak
     * capacity of the managed working buffers.
     */
    pub fn write_with_statistics(
        &self,
        extension: &str,
        destination: Destination,
    ) -> Result<ExecutionStatistics> {
        self.write_to(
            self.engine.settings().registry.encoder(extension)?,
            destination,
        )
    }

    /**
     * Run the graph with the selected encoder and the engine's execution settings.
     */
    fn write_to(
        &self,
        encoder: SelectedEncoder,
        destination: Destination,
    ) -> Result<ExecutionStatistics> {
        let settings = self.engine.settings();
        execute(
            &self.expression,
            settings.backend,
            settings.memory_limit,
            &*self.source,
            self.decoder,
            encoder,
            destination,
        )
    }
}
