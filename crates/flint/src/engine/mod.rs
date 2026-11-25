use crate::Image;
use flint_sdk::{Result, Source};
use std::{fs::File, path::Path, sync::Arc};

mod builder;
mod settings;

pub use builder::EngineBuilder;
use settings::EngineSettings;

/**
 * Codec registrations and execution defaults shared by images opened through this engine.
 */
#[derive(Clone)]
pub struct Engine(Arc<EngineSettings>);

impl Engine {
    /**
     * Create a builder with the built-in codecs and default execution settings.
     */
    pub fn builder() -> EngineBuilder {
        EngineBuilder(EngineSettings::default())
    }

    /**
     * Inspect an image file and retain its absolute path so each execution can reopen it.
     */
    pub fn open(&self, path: impl AsRef<Path>) -> Result<Image> {
        let path = std::path::absolute(path)?;
        self.open_source(move || Ok(Box::new(File::open(&path)?)))
    }

    /**
     * Inspect a stream from the source factory to select a decoder and read metadata.
     *
     * Every execution asks the factory for a fresh stream at byte zero. Each
     * stream must provide the same image data.
     */
    pub fn open_source(
        &self,
        source: impl Fn() -> Result<Source> + Send + Sync + 'static,
    ) -> Result<Image> {
        let (decoder, info) = self.0.registry.inspect(source()?)?;
        Ok(Image::from_source(self.clone(), info, decoder, source))
    }

    /**
     * Borrow the configuration shared by the engine and its images.
     */
    pub(super) fn settings(&self) -> &EngineSettings {
        &self.0
    }
}
