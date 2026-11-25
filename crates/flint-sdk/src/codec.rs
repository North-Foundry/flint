use crate::{Error, Result};
use std::io::{Read, Write};

/// Sequential input, positioned at byte zero when the decoder opens.
pub type Source = Box<dyn Read + Send>;
/// Sequential output, owned by the encoder for the duration of a session.
pub type Destination = Box<dyn Write + Send>;

/// Eight-bit, unassociated color samples. RGB rows are interleaved.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PixelFormat {
    Gray8,
    Rgb8,
}

impl PixelFormat {
    pub const fn channels(self) -> usize {
        match self {
            Self::Gray8 => 1,
            Self::Rgb8 => 3,
        }
    }
}

/// Validated nonzero dimensions and a packed row size addressable on this host.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageInfo {
    width: u32,
    height: u32,
    format: PixelFormat,
    row_bytes: usize,
}

impl ImageInfo {
    pub fn new(width: u32, height: u32, format: PixelFormat) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::Invalid("image dimensions must be nonzero".into()));
        }
        let row_bytes = usize::try_from(width)
            .ok()
            .and_then(|width| width.checked_mul(format.channels()))
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or_else(|| Error::Limit("packed row size overflow".into()))?;
        Ok(Self {
            width,
            height,
            format,
            row_bytes,
        })
    }

    pub const fn width(&self) -> u32 {
        self.width
    }
    pub const fn height(&self) -> u32 {
        self.height
    }
    pub const fn format(&self) -> PixelFormat {
        self.format
    }
    pub const fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    pub(crate) fn with_format(self, format: PixelFormat) -> Result<Self> {
        Self::new(self.width, self.height, format)
    }
}

fn row_count(info: &ImageInfo, stride: usize, bytes: usize) -> Result<usize> {
    if stride < info.row_bytes() || !bytes.is_multiple_of(stride) {
        return Err(Error::Invalid(
            "row buffer must contain whole strides at least as wide as a packed row".into(),
        ));
    }
    let count = bytes / stride;
    if count > info.height() as usize {
        return Err(Error::Invalid("row buffer exceeds image height".into()));
    }
    Ok(count)
}

/// Borrowed encoder rows, in top-to-bottom order. Each stride includes any padding.
pub struct Rows<'a> {
    info: &'a ImageInfo,
    stride: usize,
    count: usize,
    bytes: &'a [u8],
}

impl<'a> Rows<'a> {
    pub fn new(info: &'a ImageInfo, stride: usize, bytes: &'a [u8]) -> Result<Self> {
        let count = row_count(info, stride, bytes.len())?;
        Ok(Self {
            info,
            stride,
            count,
            bytes,
        })
    }

    pub const fn info(&self) -> &ImageInfo {
        self.info
    }
    pub const fn stride(&self) -> usize {
        self.stride
    }
    pub const fn len(&self) -> usize {
        self.count
    }
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Packed pixels for one row. Stride padding is excluded.
    pub fn row(&self, index: usize) -> &[u8] {
        assert!(index < self.count);
        let start = index * self.stride;
        &self.bytes[start..start + self.info.row_bytes()]
    }
}

/// Runtime-owned decoder rows, borrowed only for the current call.
pub struct RowsMut<'a> {
    info: &'a ImageInfo,
    stride: usize,
    count: usize,
    bytes: &'a mut [u8],
}

impl<'a> RowsMut<'a> {
    pub fn new(info: &'a ImageInfo, stride: usize, bytes: &'a mut [u8]) -> Result<Self> {
        let count = row_count(info, stride, bytes.len())?;
        Ok(Self {
            info,
            stride,
            count,
            bytes,
        })
    }

    pub const fn info(&self) -> &ImageInfo {
        self.info
    }
    pub const fn stride(&self) -> usize {
        self.stride
    }
    pub const fn len(&self) -> usize {
        self.count
    }
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn row(&self, index: usize) -> &[u8] {
        assert!(index < self.count);
        let start = index * self.stride;
        &self.bytes[start..start + self.info.row_bytes()]
    }

    /// Packed pixels for one row. Stride padding is excluded.
    pub fn row_mut(&mut self, index: usize) -> &mut [u8] {
        assert!(index < self.count);
        let start = index * self.stride;
        &mut self.bytes[start..start + self.info.row_bytes()]
    }
}

/// A sequential decoding session. A session can be dropped after metadata inspection.
pub trait Decoder: Send + Sized + 'static {
    const NAME: &'static str;
    const EXTENSIONS: &'static [&'static str];

    /// Inspect up to the first 64 bytes, which may be shorter for a short file.
    fn probe(header: &[u8]) -> bool;
    /// Open at the start of a stream and read its metadata.
    fn open(source: Source) -> Result<Self>;
    fn info(&self) -> &ImageInfo;

    /// Fill a prefix of the provided rows and return its length. With a nonempty
    /// buffer, zero means end of image. A truncated image must return an error.
    fn read_rows(&mut self, output: RowsMut<'_>) -> Result<usize>;

    /// Validate end-of-stream data after exactly the declared height has been read.
    /// Implement this when the container has a footer, checksum, or other final validation.
    fn finish(self) -> Result<()> {
        Ok(())
    }
}

/// A sequential encoding session. Successful completion requires calling `finish`.
pub trait Encoder: Send + Sized + 'static {
    const NAME: &'static str;
    const EXTENSIONS: &'static [&'static str];
    type Options: Default;

    fn open(destination: Destination, info: &ImageInfo, options: Self::Options) -> Result<Self>;
    /// Consume packed rows before returning. Reject rows that exceed the declared height.
    fn write_rows(&mut self, rows: Rows<'_>) -> Result<()>;
    /// Reject incomplete images and report trailer or flush errors.
    fn finish(self) -> Result<()>;
}
