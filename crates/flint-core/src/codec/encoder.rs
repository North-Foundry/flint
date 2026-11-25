/*!
 * Encoder handles and boxed encoder sessions.
 */

use flint_sdk::{Destination, Encoder, Error, ImageInformation, Result, Rows};

/**
 * Adapt an SDK encoder for use through a boxed session.
 */
pub(crate) trait EncoderSession {
    /**
     * Consume packed rows before returning.
     */
    fn write_rows(&mut self, rows: Rows<'_>) -> Result<()>;

    /**
     * Validate the complete image and report trailer or flush errors.
     */
    fn finish(self: Box<Self>) -> Result<()>;
}

impl<T: Encoder> EncoderSession for T {
    fn write_rows(&mut self, rows: Rows<'_>) -> Result<()> {
        Encoder::write_rows(self, rows)
    }

    fn finish(self: Box<Self>) -> Result<()> {
        Encoder::finish(*self)
    }
}

/**
 * Opens an encoder with default options and boxes its session.
 */
type SessionOpener = fn(Destination, &ImageInformation) -> Result<Box<dyn EncoderSession>>;

/**
 * Open a concrete encoder with its default options and box its session.
 */
fn open_session<T: Encoder>(
    destination: Destination,
    info: &ImageInformation,
) -> Result<Box<dyn EncoderSession>> {
    Ok(Box::new(T::open(destination, info, T::Options::default())?))
}

/**
 * A registered encoder identified by its extensions and session factory.
 */
#[derive(Clone, Copy)]
pub struct SelectedEncoder {
    extensions: &'static [&'static str],
    open: SessionOpener,
}

impl SelectedEncoder {
    /**
     * Create the handle stored in the registry for an encoder type.
     */
    pub(super) fn new<T: Encoder>() -> Self {
        Self {
            extensions: T::EXTENSIONS,
            open: open_session::<T>,
        }
    }

    /**
     * Open an encoding session using the graph's output metadata.
     */
    pub(crate) fn session(
        self,
        destination: Destination,
        info: &ImageInformation,
    ) -> Result<Box<dyn EncoderSession>> {
        (self.open)(destination, info)
    }

    /**
     * Check whether this encoder accepts the extension, ignoring ASCII case.
     */
    fn supports(self, extension: &str) -> bool {
        self.extensions
            .iter()
            .any(|candidate| extension.eq_ignore_ascii_case(candidate))
    }
}

/**
 * Select the most recently registered encoder that accepts the extension.
 */
pub(crate) fn select(encoders: &[SelectedEncoder], extension: &str) -> Result<SelectedEncoder> {
    encoders
        .iter()
        .rev()
        .find(|encoder| encoder.supports(extension))
        .copied()
        .ok_or_else(|| Error::Unsupported(format!("no encoder for extension {extension:?}")))
}
