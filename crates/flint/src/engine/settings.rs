use crate::{Backend, Optimization};
use flint_core::Registry;

const DEFAULT_MEMORY_LIMIT: usize = 64 * 1024 * 1024; // 64MiB

/**
 * Codec registry and execution defaults shared by images opened with an engine.
 */
pub(crate) struct EngineSettings {
    pub(crate) registry: Registry,
    pub(crate) memory_limit: usize,
    pub(crate) backend: Backend,
    pub(crate) optimization: Optimization,
}

impl Default for EngineSettings {
    /**
     * Register the codecs enabled at compile time and select a 64 MiB memory limit
     * with automatic backend selection and pointwise fusion.
     */
    fn default() -> Self {
        #[allow(unused_mut)]
        let mut registry = Registry::default();
        #[cfg(feature = "png")]
        {
            registry.register_decoder::<flint_png::PngDecoder>();
            registry.register_encoder::<flint_png::PngEncoder>();
        }
        #[cfg(feature = "jpeg")]
        registry.register_encoder::<flint_jpeg::JpegEncoder>();

        Self {
            registry,
            memory_limit: DEFAULT_MEMORY_LIMIT,
            backend: Backend::Auto,
            optimization: Optimization::Auto,
        }
    }
}
