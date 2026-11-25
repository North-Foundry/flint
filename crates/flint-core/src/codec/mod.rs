/*!
 * Decoder and encoder registration for the streaming runtime.
 *
 * The registry stores lightweight handles to codec factories. The session
 * adapters and selection rules live in the decoder and encoder submodules.
 */

mod decoder;
mod encoder;

pub(crate) use decoder::DecoderSession;
pub use decoder::SelectedDecoder;
pub use encoder::SelectedEncoder;

use flint_sdk::{Decoder, Encoder, ImageInformation, Result, Source};

/**
 * The decoders and encoders available to an engine.
 */
#[derive(Default)]
pub struct Registry {
    decoders: Vec<SelectedDecoder>,
    encoders: Vec<SelectedEncoder>,
}

impl Registry {
    /**
     * Register a decoder type. Later registrations take precedence during probing.
     */
    pub fn register_decoder<T: Decoder>(&mut self) {
        self.decoders.push(SelectedDecoder::new::<T>());
    }

    /**
     * Register an encoder type. Later registrations take precedence for an extension.
     */
    pub fn register_encoder<T: Encoder>(&mut self) {
        self.encoders.push(SelectedEncoder::new::<T>());
    }

    /**
     * Match the source prefix against registered decoders and read the image metadata.
     */
    pub fn inspect(&self, source: Source) -> Result<(SelectedDecoder, ImageInformation)> {
        decoder::select(&self.decoders, source)
    }

    /**
     * Select the encoder registered for an extension, ignoring ASCII case.
     */
    pub fn encoder(&self, extension: &str) -> Result<SelectedEncoder> {
        encoder::select(&self.encoders, extension)
    }
}
