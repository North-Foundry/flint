use crate::{Engine, ExecutionPlan, ExecutionStats};
use flint_core::{SelectedDecoder, SelectedEncoder, execute};
use flint_sdk::{Destination, Error, ImageExpr, ImageInfo, Operation, Result, Source};
use std::{
    path::Path,
    sync::{Arc, OnceLock},
};

/// A reusable symbolic image. Cloning shares the source and expression graph.
#[derive(Clone)]
pub struct Image {
    engine: Engine,
    expr: ImageExpr,
    decoder: SelectedDecoder,
    source: Arc<dyn Fn() -> Result<Source> + Send + Sync>,
}

impl Image {
    pub(super) fn from_source(
        engine: Engine,
        info: ImageInfo,
        decoder: SelectedDecoder,
        source: impl Fn() -> Result<Source> + Send + Sync + 'static,
    ) -> Self {
        Self {
            engine,
            expr: ImageExpr::source(info),
            decoder,
            source: Arc::new(source),
        }
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        static DEFAULT: OnceLock<Engine> = OnceLock::new();
        DEFAULT
            .get_or_init(|| Engine::builder().build().expect("valid default engine"))
            .open(path)
    }

    pub fn info(&self) -> &ImageInfo {
        self.expr.info()
    }

    pub fn apply(mut self, operation: impl Operation) -> Result<Self> {
        let expr = operation.apply(self.expr.clone())?;
        if !self.expr.shares_source(&expr) {
            return Err(Error::Invalid(
                "operation returned a graph with a different source".into(),
            ));
        }
        self.expr = expr;
        Ok(self)
    }

    pub fn explain(&self) -> Result<ExecutionPlan> {
        ExecutionPlan::analyze(
            &self.expr,
            self.decoder.name(),
            self.engine.settings().backend,
        )
    }

    /// Stream to a private file beside the destination and replace it after success.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.save_with_stats(path).map(drop)
    }

    /// Save the image and report the backend and measured managed buffer capacity.
    pub fn save_with_stats(&self, path: impl AsRef<Path>) -> Result<ExecutionStats> {
        let path = std::path::absolute(path)?;
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| {
                Error::Unsupported("destination must have a recognized extension".into())
            })?;
        let encoder = self.engine.settings().registry.encoder(extension)?;
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let temporary = tempfile::NamedTempFile::new_in(parent)?;
        let stats = self.write_to(encoder, Box::new(temporary.reopen()?))?;
        temporary.persist(path).map_err(|error| error.error)?;
        Ok(stats)
    }

    /// Stream to a sequential destination. An execution error may leave partial output.
    pub fn write(&self, extension: &str, destination: Destination) -> Result<()> {
        self.write_with_stats(extension, destination).map(drop)
    }

    /// Write to a sequential destination and report measured managed buffer capacity.
    pub fn write_with_stats(
        &self,
        extension: &str,
        destination: Destination,
    ) -> Result<ExecutionStats> {
        self.write_to(
            self.engine.settings().registry.encoder(extension)?,
            destination,
        )
    }

    fn write_to(
        &self,
        encoder: SelectedEncoder,
        destination: Destination,
    ) -> Result<ExecutionStats> {
        let settings = self.engine.settings();
        execute(
            &self.expr,
            settings.backend,
            settings.memory_limit,
            &*self.source,
            self.decoder,
            encoder,
            destination,
        )
    }
}
