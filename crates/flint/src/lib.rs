//! Open an image lazily, compose operations, and save with automatic codec selection.

use flint_core::{Registry, SelectedDecoder, SelectedEncoder, execute};
use flint_sdk::{Decoder, Destination, Encoder, ImageExpr, ImageInfo, Operation, Source};
use std::{
    fs::File,
    path::Path,
    sync::{Arc, OnceLock},
};

pub use flint_core::{Backend, ExecutionPlan, ExecutionStats, StagePlan};
pub use flint_sdk as sdk;
pub use flint_sdk::{Error, Result};

const DEFAULT_MEMORY_LIMIT: usize = 64 * 1024 * 1024;

/// Codec registrations and execution defaults shared by images opened through this engine.
#[derive(Clone)]
pub struct Engine(Arc<EngineSettings>);

struct EngineSettings {
    registry: Registry,
    memory_limit: usize,
    backend: Backend,
}

pub struct EngineBuilder(EngineSettings);

impl Engine {
    pub fn builder() -> EngineBuilder {
        #[allow(unused_mut)]
        let mut registry = Registry::default();
        #[cfg(feature = "png")]
        {
            registry.register_decoder::<flint_png::PngDecoder>();
            registry.register_encoder::<flint_png::PngEncoder>();
        }
        #[cfg(feature = "jpeg")]
        registry.register_encoder::<flint_jpeg::JpegEncoder>();

        EngineBuilder(EngineSettings {
            registry,
            memory_limit: DEFAULT_MEMORY_LIMIT,
            backend: Backend::Auto,
        })
    }

    pub fn open(&self, path: impl AsRef<Path>) -> Result<Image> {
        let path = std::path::absolute(path)?;
        self.open_source(move || Ok(Box::new(File::open(&path)?)))
    }

    /// Inspect a new stream now. Every execution asks the factory for a fresh stream
    /// at byte zero. The factory must provide unchanged image data across openings.
    pub fn open_source(
        &self,
        source: impl Fn() -> Result<Source> + Send + Sync + 'static,
    ) -> Result<Image> {
        let (decoder, info) = self.0.registry.inspect(source()?)?;
        Ok(Image {
            engine: self.clone(),
            expr: ImageExpr::source(info),
            decoder,
            source: Arc::new(source),
        })
    }
}

impl EngineBuilder {
    /// Set the cap on reusable execution pixel buffers and evaluator scratch space.
    pub fn memory_limit(mut self, bytes: usize) -> Self {
        self.0.memory_limit = bytes;
        self
    }

    pub fn backend(mut self, backend: Backend) -> Self {
        self.0.backend = backend;
        self
    }

    /// Add a decoder factory. The most recently registered matching decoder wins.
    pub fn register_decoder<T: Decoder>(mut self) -> Self {
        self.0.registry.register_decoder::<T>();
        self
    }

    /// Add an encoder factory. The most recently registered matching extension wins.
    pub fn register_encoder<T: Encoder>(mut self) -> Self {
        self.0.registry.register_encoder::<T>();
        self
    }

    pub fn build(self) -> Result<Engine> {
        if self.0.memory_limit == 0 {
            return Err(Error::Limit("managed memory limit must be nonzero".into()));
        }
        Ok(Engine(Arc::new(self.0)))
    }
}

/// A reusable symbolic image. Cloning shares the source and expression graph.
#[derive(Clone)]
pub struct Image {
    engine: Engine,
    expr: ImageExpr,
    decoder: SelectedDecoder,
    source: Arc<dyn Fn() -> Result<Source> + Send + Sync>,
}

impl Image {
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
        ExecutionPlan::analyze(&self.expr, self.decoder.name(), self.engine.0.backend)
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
        let encoder = self.engine.0.registry.encoder(extension)?;
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
        self.write_to(self.engine.0.registry.encoder(extension)?, destination)
    }

    fn write_to(
        &self,
        encoder: SelectedEncoder,
        destination: Destination,
    ) -> Result<ExecutionStats> {
        let settings = &self.engine.0;
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
