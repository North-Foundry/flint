use super::{Engine, EngineSettings};
use crate::{Backend, Optimization};
use flint_sdk::{Decoder, Encoder, Error, Result};
use std::sync::Arc;

pub struct EngineBuilder(pub(super) EngineSettings);

impl EngineBuilder {
    /**
     * Set the limit in bytes for reusable pixel buffers and evaluator scratch space.
     */
    pub fn memory_limit(mut self, bytes: usize) -> Self {
        self.0.memory_limit = bytes;
        self
    }

    /**
     * Set the preferred execution backend. The runtime resolves this choice when planning.
     */
    pub fn backend(mut self, backend: Backend) -> Self {
        self.0.backend = backend;
        self
    }

    /**
     * Select pointwise kernel fusion independently of the execution backend.
     * The default is automatic.
     */
    pub fn optimization(mut self, optimization: Optimization) -> Self {
        self.0.optimization = optimization;
        self
    }

    /**
     * Add a decoder factory. The most recently registered matching decoder wins.
     */
    pub fn register_decoder<T: Decoder>(mut self) -> Self {
        self.0.registry.register_decoder::<T>();
        self
    }

    /**
     * Add an encoder factory. For a given extension, the most recently registered
     * matching encoder wins.
     */
    pub fn register_encoder<T: Encoder>(mut self) -> Self {
        self.0.registry.register_encoder::<T>();
        self
    }

    /**
     * Consume the builder and create an engine with shared settings.
     *
     * # Errors
     *
     * Returns [`Error::Limit`] if the memory limit is zero.
     */
    pub fn build(self) -> Result<Engine> {
        if self.0.memory_limit == 0 {
            return Err(Error::Limit("managed memory limit must be nonzero".into()));
        }
        Ok(Engine(Arc::new(self.0)))
    }
}
