use std::io::{self, BufRead, BufReader};

use flint_sdk::{Decoder, Error, ImageInformation, PixelFormat, Result, RowsMut, Source};
use png::chunk::{IDAT, IEND, IHDR, acTL, eXIf, fcTL, fdAT};
use png::{
    BitDepth, ColorType, DecodeOptions, Decoded, Info, StreamingDecoder, UnfilterBuf,
    UnfilterRegion,
};

use crate::filter::unfilter;

const SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";
const DEFLATE_HISTORY: usize = 32 * 1024;
const DEFLATE_MAX_MATCH: usize = 258;
const MAX_ROW_BYTES: usize = 16 * 1024 * 1024;
const MAX_EXIF_BYTES: u32 = 1024 * 1024;

/**
 * Reads non-interlaced Gray8 and RGB8 PNGs. Finishing validates the zlib
 * trailer, PNG chunk CRCs and IEND. Packed rows are capped at 16 MiB and EXIF at 1 MiB.
 */
pub struct PngDecoder {
    stream: PngStream,
    info: ImageInformation,
    raw: Vec<u8>,
    region: UnfilterRegion,
    row_start: usize,
    previous: Vec<u8>,
    remaining_bytes: u64,
    remaining_rows: usize,
    image_data_complete: bool,
    failed: bool,
}

impl Decoder for PngDecoder {
    const NAME: &'static str = "png";
    const EXTENSIONS: &'static [&'static str] = &["png"];

    fn probe(header: &[u8]) -> bool {
        header.starts_with(SIGNATURE)
    }

    /**
     * Read and validate the PNG header before allocating pixel row buffers.
     */
    fn open(source: Source) -> Result<Self> {
        let mut stream = PngStream::new(source);
        while !matches!(stream.next(None)?, Decoded::ChunkComplete(IHDR)) {}
        let header = stream.info()?;
        let format = pixel_format(header)?;
        if header.width > i32::MAX as u32 || header.height > i32::MAX as u32 {
            return Err(Error::Invalid("PNG dimensions must fit in 31 bits".into()));
        }
        let info = ImageInformation::new(header.width, header.height, format)?;
        if info.row_bytes() > MAX_ROW_BYTES {
            return Err(Error::Limit("PNG packed row exceeds 16 MiB".into()));
        }

        loop {
            match stream.next(None)? {
                Decoded::ChunkBegin(_, IDAT) => break,
                Decoded::ChunkComplete(IEND) => {
                    return Err(Error::Codec("missing PNG IDAT".into()));
                }
                _ => (),
            }
        }
        check_metadata(stream.info()?)?;

        let raw_row = info.row_bytes() + 1;
        let remaining_bytes = raw_row as u64 * u64::from(info.height());
        Ok(Self {
            stream,
            info,
            raw: Vec::new(),
            region: UnfilterRegion::default(),
            row_start: 0,
            previous: Vec::new(),
            remaining_bytes,
            remaining_rows: info.height() as usize,
            image_data_complete: false,
            failed: false,
        })
    }

    fn info(&self) -> &ImageInformation {
        &self.info
    }

    fn read_rows(&mut self, mut output: RowsMut<'_>) -> Result<usize> {
        if self.failed {
            return Err(Error::Codec("PNG decoder has failed".into()));
        }
        if output.info() != &self.info {
            return Err(Error::Invalid(
                "PNG output rows do not match image info".into(),
            ));
        }

        let count = output.len().min(self.remaining_rows);
        if count == 0 {
            return Ok(0);
        }
        self.failed = true;
        if self.raw.is_empty() {
            self.allocate_rows()?;
        }
        for i in 0..count {
            self.read_row(output.row_mut(i))?;
            self.remaining_rows -= 1;
        }
        self.failed = false;
        Ok(count)
    }

    fn finish(mut self) -> Result<()> {
        if self.failed {
            return Err(Error::Codec("PNG decoder has failed".into()));
        }
        if self.remaining_rows != 0 {
            return Err(Error::Invalid(
                "PNG rows must be read before finishing".into(),
            ));
        }

        // A raw row can be available while zlib still expects its trailer.
        while !self.image_data_complete {
            self.inflate()?;
        }
        while !matches!(self.stream.next(None)?, Decoded::ChunkComplete(IEND)) {}
        check_metadata(self.stream.info()?)
    }
}

impl PngDecoder {
    fn allocate_rows(&mut self) -> Result<()> {
        // Keep one row and up to 32 KiB of raw history, with another 32 KiB
        // to amortize compaction. Always allow space for a complete queued match.
        let full_window = self.info.row_bytes() + 1 + 2 * DEFLATE_HISTORY + DEFLATE_MAX_MATCH;
        let window =
            (full_window as u64).min(self.remaining_bytes + DEFLATE_MAX_MATCH as u64) as usize;
        self.raw = zeroed(window)?;
        self.previous = zeroed(self.info.row_bytes())?;
        Ok(())
    }

    fn read_row(&mut self, output: &mut [u8]) -> Result<()> {
        let raw_row = output.len() + 1;
        while self.region.filled - self.row_start < raw_row {
            self.inflate()?;
        }
        let end = self.row_start + raw_row;
        let raw = &self.raw[self.row_start..end];
        // Only read the raw bytes. They may still be needed as DEFLATE history.
        unfilter(
            raw[0],
            &raw[1..],
            &self.previous,
            self.info.format(),
            output,
        )?;
        self.previous.copy_from_slice(output);
        self.row_start = end;
        Ok(())
    }

    fn inflate(&mut self) -> Result<()> {
        if self.image_data_complete {
            return Err(missing_rows());
        }
        if self.raw.len() - self.region.filled < DEFLATE_MAX_MATCH {
            let discard = self.row_start.min(self.region.available);
            if discard == 0 {
                return Err(Error::Codec("PNG inflate window made no progress".into()));
            }
            self.raw.copy_within(discard..self.region.filled, 0);
            self.region.filled -= discard;
            self.region.available -= discard;
            self.row_start -= discard;
        }

        // Spare output space prevents png from reporting premature completion.
        // A whole match also avoids fdeflate 0.3.7's early return before its
        // checksum update when a previously queued match only partly fits.
        let before = self.region.filled;
        let event = self
            .stream
            .next(Some(&mut self.region.as_buf(&mut self.raw)))?;
        let produced = (self.region.filled - before) as u64;
        self.remaining_bytes = self
            .remaining_bytes
            .checked_sub(produced)
            .ok_or_else(|| Error::Codec("PNG contains excess decompressed image data".into()))?;
        if matches!(event, Decoded::ImageDataFlushed) {
            if self.remaining_bytes != 0 {
                return Err(missing_rows());
            }
            self.image_data_complete = true;
        }
        Ok(())
    }
}

struct PngStream {
    decoder: StreamingDecoder,
    input: BufReader<Source>,
}

impl PngStream {
    fn new(source: Source) -> Self {
        let mut options = DecodeOptions::default();
        options.set_ignore_checksums(false);
        options.set_skip_ancillary_crc_failures(false);
        options.set_ignore_text_chunk(true);
        options.set_ignore_iccp_chunk(true);
        Self {
            decoder: StreamingDecoder::new_with_options(options),
            input: BufReader::new(source),
        }
    }

    fn info(&self) -> Result<&Info<'static>> {
        self.decoder
            .info()
            .ok_or_else(|| Error::Codec("missing PNG IHDR".into()))
    }

    fn next(&mut self, image_data: Option<&mut UnfilterBuf<'_>>) -> Result<Decoded> {
        let (consumed, event) = loop {
            match self.input.fill_buf() {
                Ok([]) => {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "truncated PNG stream",
                    )
                    .into());
                }
                Ok(bytes) => {
                    break self
                        .decoder
                        .update(bytes, image_data)
                        .map_err(decode_error)?;
                }
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error.into()),
            }
        };
        self.input.consume(consumed);
        if let Decoded::ChunkBegin(length, kind) = event {
            if length > i32::MAX as u32 {
                return Err(Error::Limit("PNG chunk length exceeds 31 bits".into()));
            }
            // ChunkBegin returns before payload allocation. In png 0.18.1,
            // disabled text/ICC and unknown ancillary chunks are skipped, IDAT
            // is streamed, and eXIf is the only remaining variable-size allocation.
            if kind == eXIf && length > MAX_EXIF_BYTES {
                return Err(Error::Limit("PNG EXIF chunk exceeds 1 MiB".into()));
            }
            if [acTL, fcTL, fdAT].contains(&kind) {
                return Err(Error::Unsupported("animated PNG".into()));
            }
        }
        Ok(event)
    }
}

fn zeroed(bytes: usize) -> Result<Vec<u8>> {
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(bytes)
        .map_err(|_| Error::Limit("could not allocate PNG row buffer".into()))?;
    buffer.resize(bytes, 0);
    Ok(buffer)
}

fn pixel_format(info: &Info<'_>) -> Result<PixelFormat> {
    if info.interlaced {
        return Err(Error::Unsupported("interlaced PNG".into()));
    }
    match (info.color_type, info.bit_depth) {
        (ColorType::Grayscale, BitDepth::Eight) => Ok(PixelFormat::Gray8),
        (ColorType::Rgb, BitDepth::Eight) => Ok(PixelFormat::Rgb8),
        (color, depth) => Err(Error::Unsupported(format!(
            "PNG {color:?} with {depth:?} bit depth"
        ))),
    }
}

fn check_metadata(info: &Info<'_>) -> Result<()> {
    if info.trns.is_some() {
        return Err(Error::Unsupported("PNG transparency (tRNS)".into()));
    }
    Ok(())
}

fn missing_rows() -> Error {
    io::Error::new(io::ErrorKind::UnexpectedEof, "missing PNG rows").into()
}

fn decode_error(error: png::DecodingError) -> Error {
    match error {
        png::DecodingError::IoError(error) => error.into(),
        png::DecodingError::LimitsExceeded => Error::Limit("PNG decoder limits exceeded".into()),
        other => Error::Codec(format!("PNG: {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_rows_on_first_nonempty_read() {
        let mut bytes = Vec::new();
        let mut writer = png::Encoder::new(&mut bytes, 1, 1).write_header().unwrap();
        writer.write_image_data(&[42]).unwrap();
        writer.finish().unwrap();

        let mut decoder = PngDecoder::open(Box::new(io::Cursor::new(bytes))).unwrap();
        let info = *decoder.info();
        assert_eq!(
            (decoder.raw.capacity(), decoder.previous.capacity()),
            (0, 0)
        );
        let empty = RowsMut::new(&info, 1, &mut []).unwrap();
        assert_eq!(decoder.read_rows(empty).unwrap(), 0);
        assert_eq!(
            (decoder.raw.capacity(), decoder.previous.capacity()),
            (0, 0)
        );

        let mut pixel = [0];
        let output = RowsMut::new(&info, 1, &mut pixel).unwrap();
        assert_eq!(decoder.read_rows(output).unwrap(), 1);
        assert_eq!(pixel, [42]);
        decoder.finish().unwrap();
    }
}
