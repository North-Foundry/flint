/*!
 * Decoder handles, probing, and boxed decoder sessions.
 */

use flint_sdk::{Decoder, Error, ImageInformation, Result, RowsMut, Source};
use std::io::{Cursor, Read};

/**
 * Adapt an SDK decoder for use through a boxed session.
 */
pub(crate) trait DecoderSession {
    /**
     * Return the metadata reported by the underlying decoder.
     */
    fn info(&self) -> &ImageInformation;

    /**
     * Fill a prefix with the provided rows and return the number of rows written.
     */
    fn read_rows(&mut self, output: RowsMut<'_>) -> Result<usize>;

    /**
     * Validate the end of the stream after the declared height has been read.
     */
    fn finish(self: Box<Self>) -> Result<()>;
}

impl<T: Decoder> DecoderSession for T {
    fn info(&self) -> &ImageInformation {
        Decoder::info(self)
    }

    fn read_rows(&mut self, output: RowsMut<'_>) -> Result<usize> {
        Decoder::read_rows(self, output)
    }

    fn finish(self: Box<Self>) -> Result<()> {
        Decoder::finish(*self)
    }
}

/**
 * Opens a decoder from the start of a source and boxes its session.
 */
type SessionOpener = fn(Source) -> Result<Box<dyn DecoderSession>>;

/**
 * Open a concrete decoder and box it for use through a session trait object.
 */
fn open_session<T: Decoder>(source: Source) -> Result<Box<dyn DecoderSession>> {
    Ok(Box::new(T::open(source)?))
}

/**
 * A registered decoder identified by its signature probe and session factory.
 */
#[derive(Clone, Copy)]
pub struct SelectedDecoder {
    name: &'static str,
    probe: fn(&[u8]) -> bool,
    open: SessionOpener,
}

impl SelectedDecoder {
    /**
     * Create the handle stored in the registry for a decoder type.
     */
    pub(super) fn new<T: Decoder>() -> Self {
        Self {
            name: T::NAME,
            probe: T::probe,
            open: open_session::<T>,
        }
    }

    /**
     * Return the codec name used in execution plans.
     */
    pub fn name(self) -> &'static str {
        self.name
    }

    /**
     * Open a decoding session at byte zero of the source.
     */
    pub(crate) fn session(self, source: Source) -> Result<Box<dyn DecoderSession>> {
        (self.open)(source)
    }

    /**
     * Check whether the source prefix matches this decoder's signature.
     */
    fn matches(self, prefix: &[u8]) -> bool {
        (self.probe)(prefix)
    }
}

/**
 * Read the detection prefix, choose the most recently registered matching
 * decoder, and read the image metadata from a fresh session.
 */
pub(crate) fn select(
    decoders: &[SelectedDecoder],
    mut source: Source,
) -> Result<(SelectedDecoder, ImageInformation)> {
    let mut prefix = [0; 64];
    let mut length = 0;
    while length < prefix.len() {
        match source.read(&mut prefix[length..]) {
            Ok(0) => break,
            Ok(count) => length += count,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error.into()),
        }
    }
    let decoder = decoders
        .iter()
        .rev()
        .find(|decoder| decoder.matches(&prefix[..length]))
        .copied()
        .ok_or_else(|| Error::Unsupported("unrecognized input format".into()))?;
    let session = decoder.session(Box::new(
        Cursor::new(prefix).take(length as u64).chain(source),
    ))?;
    Ok((decoder, *session.info()))
}
