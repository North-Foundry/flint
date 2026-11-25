use std::io::{self, Write};
use std::sync::{Arc, Mutex};

use flint_sdk::{Destination, Encoder, Error, ImageInformation, PixelFormat, Result, Rows};
use png::{BitDepth, ColorType, EncodingError, StreamWriter};

/**
 * Encodes Gray8 and RGB8 rows to a non-interlaced PNG stream.
 */
pub struct PngEncoder {
    writer: StreamWriter<'static, CheckedDestination>,
    destination: CheckedDestination,
    info: ImageInformation,
    remaining_rows: usize,
    failed: bool,
}

impl Encoder for PngEncoder {
    const NAME: &'static str = "png";
    const EXTENSIONS: &'static [&'static str] = &["png"];
    type Options = ();

    /**
     * Check the encoder's dimension limits and write the PNG header.
     */
    fn open(destination: Destination, info: &ImageInformation, (): Self::Options) -> Result<Self> {
        let remaining_rows = info.height() as usize;
        // PNG dimensions are 31-bit, and png's StreamWriter counts packed bytes in usize.
        if info.width() > i32::MAX as u32
            || info.height() > i32::MAX as u32
            || info.row_bytes().checked_mul(remaining_rows).is_none()
        {
            return Err(Error::Limit("PNG dimensions exceed encoder limits".into()));
        }

        let destination = CheckedDestination(Arc::new(Mutex::new(Output {
            writer: destination,
            error: None,
        })));
        let mut encoder = png::Encoder::new(destination.clone(), info.width(), info.height());
        encoder.set_color(match info.format() {
            PixelFormat::Gray8 => ColorType::Grayscale,
            PixelFormat::Rgb8 => ColorType::Rgb,
        });
        encoder.set_depth(BitDepth::Eight);
        encoder.validate_sequence(true);

        let writer = encoder
            .write_header()
            .and_then(|writer| writer.into_stream_writer());
        destination.check_error()?;
        let writer = writer.map_err(encode_error)?;
        Ok(Self {
            writer,
            destination,
            info: *info,
            remaining_rows,
            failed: false,
        })
    }

    fn write_rows(&mut self, rows: Rows<'_>) -> Result<()> {
        if self.failed {
            return Err(Error::Codec("PNG encoder has failed".into()));
        }
        if rows.info() != &self.info || rows.len() > self.remaining_rows {
            return Err(Error::Invalid(
                "PNG input rows do not match remaining image rows".into(),
            ));
        }
        for i in 0..rows.len() {
            self.failed = true;
            if let Err(error) = self.writer.write_all(rows.row(i)) {
                self.destination.check_error()?;
                return Err(error.into());
            }
            self.remaining_rows -= 1;
            self.failed = false;
        }
        Ok(())
    }

    fn finish(self) -> Result<()> {
        if self.failed {
            return Err(Error::Codec("PNG encoder has failed".into()));
        }
        if self.remaining_rows != 0 {
            return Err(Error::Invalid("incomplete PNG image".into()));
        }
        let result = self.writer.finish();
        self.destination.check_error()?;
        result.map_err(encode_error)?;
        self.destination.flush_output()?;
        Ok(())
    }
}

fn encode_error(error: EncodingError) -> Error {
    match error {
        EncodingError::IoError(error) => error.into(),
        EncodingError::LimitsExceeded => Error::Limit("PNG encoder limits exceeded".into()),
        other => Error::Codec(format!("PNG: {other}")),
    }
}

// png's owning StreamWriter writes IEND in its inner Writer's Drop, which discards
// I/O errors. Retain the first failure so finish can report it, then flush the sink.
#[derive(Clone)]
struct CheckedDestination(Arc<Mutex<Output>>);

struct Output {
    writer: Destination,
    error: Option<io::Error>,
}

impl CheckedDestination {
    fn flush_output(&self) -> io::Result<()> {
        self.0
            .lock()
            .map_err(|_| io::Error::other("PNG destination panicked"))?
            .writer
            .flush()
    }

    fn check_error(&self) -> io::Result<()> {
        let mut output = self
            .0
            .lock()
            .map_err(|_| io::Error::other("PNG destination panicked"))?;
        match &mut output.error {
            Some(error) => Err(std::mem::replace(error, error.kind().into())),
            None => Ok(()),
        }
    }

    fn with_writer(
        &self,
        operation: impl FnOnce(&mut Destination) -> io::Result<()>,
    ) -> io::Result<()> {
        let mut output = self
            .0
            .lock()
            .map_err(|_| io::Error::other("PNG destination panicked"))?;
        if let Some(error) = &output.error {
            return Err(error.kind().into());
        }
        operation(&mut output.writer).map_err(|error| {
            let kind = error.kind();
            output.error = Some(error);
            kind.into()
        })
    }
}

impl Write for CheckedDestination {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.with_writer(|writer| writer.write_all(bytes))?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.with_writer(Write::flush)
    }
}
