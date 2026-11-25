//! Single-scan baseline JPEG encoding for Flint.

use std::any::Any;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::io;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::ptr::NonNull;
use std::slice;

use flint_sdk::{Destination, Encoder, Error, ImageInfo, PixelFormat, Result, Rows};
use mozjpeg_sys as _; // Link the bundled library used by the C bridge.

/// JPEG quality must be in 1..=100. The default is 85.
#[derive(Clone, Copy, Debug)]
pub struct JpegOptions {
    pub quality: u8,
}

impl Default for JpegOptions {
    fn default() -> Self {
        Self { quality: 85 }
    }
}

/// Encodes borrowed scanlines as baseline sequential JPEG. RGB8 is converted
/// to YCbCr with 4:2:0 chroma sampling. Gray8 uses a single luminance component.
pub struct JpegEncoder {
    session: Session,
    destination: Destination,
    info: ImageInfo,
    remaining_rows: usize,
    failed: bool,
}

impl Encoder for JpegEncoder {
    const NAME: &'static str = "jpeg";
    const EXTENSIONS: &'static [&'static str] = &["jpg", "jpeg"];
    type Options = JpegOptions;

    fn open(destination: Destination, info: &ImageInfo, options: Self::Options) -> Result<Self> {
        if !(1..=100).contains(&options.quality) {
            return Err(Error::Invalid(
                "JPEG quality must be between 1 and 100".into(),
            ));
        }
        // SAFETY: This function returns a build-time constant and dereferences nothing.
        let max_dimension = unsafe { flint_jpeg_max_dimension() };
        if info.width() > max_dimension || info.height() > max_dimension {
            return Err(Error::Limit(format!(
                "JPEG dimensions exceed {max_dimension}"
            )));
        }

        let components = match info.format() {
            PixelFormat::Gray8 => 1,
            PixelFormat::Rgb8 => 3,
        };
        let mut encoder = Self {
            session: Session::new()?,
            destination,
            info: *info,
            remaining_rows: info.height() as usize,
            failed: false,
        };
        // SAFETY: The session and output are valid for the synchronous start call.
        encoder.run(|session, output| unsafe {
            flint_jpeg_start(
                session,
                output,
                info.width(),
                info.height(),
                components,
                options.quality.into(),
            )
        })?;
        Ok(encoder)
    }

    fn write_rows(&mut self, rows: Rows<'_>) -> Result<()> {
        if self.failed {
            return Err(Error::Codec("JPEG encoder has failed".into()));
        }
        if rows.info() != &self.info || rows.len() > self.remaining_rows {
            return Err(Error::Invalid(
                "JPEG input rows do not match remaining image rows".into(),
            ));
        }
        for i in 0..rows.len() {
            let row = rows.row(i);
            // SAFETY: The SDK row has the negotiated packed length and the C
            // encoder reads it only during this synchronous call.
            self.run(|session, output| unsafe {
                flint_jpeg_write_row(session, output, row.as_ptr())
            })?;
            self.remaining_rows -= 1;
        }
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        if self.failed {
            return Err(Error::Codec("JPEG encoder has failed".into()));
        }
        if self.remaining_rows != 0 {
            return Err(Error::Invalid("incomplete JPEG image".into()));
        }
        // SAFETY: The session is live and all scanlines have been accepted.
        self.run(|session, output| unsafe { flint_jpeg_finish(session, output) })?;
        self.destination.flush()?;
        Ok(())
    }
}

impl JpegEncoder {
    fn run(&mut self, operation: impl FnOnce(NonNull<c_void>, *mut c_void) -> c_int) -> Result<()> {
        let mut output = Output {
            writer: &mut self.destination,
            failure: None,
        };
        self.failed = true;
        let success = operation(self.session.0, (&raw mut output).cast());
        match output.failure {
            Some(WriteFailure::Io(error)) => return Err(error.into()),
            Some(WriteFailure::Panic(payload)) => resume_unwind(payload),
            None => (),
        }
        if success == 0 {
            // SAFETY: The C bridge keeps a NUL-terminated error message in the live session.
            let message = unsafe { CStr::from_ptr(flint_jpeg_error(self.session.0)) };
            return Err(Error::Codec(format!("JPEG: {}", message.to_string_lossy())));
        }
        self.failed = false;
        Ok(())
    }
}

struct Session(NonNull<c_void>);

// SAFETY: The allocation is uniquely owned and stays at a fixed address. C
// retains no Rust references between calls and has no thread-affine state.
unsafe impl Send for Session {}

impl Session {
    fn new() -> Result<Self> {
        // SAFETY: The callback uses the C ABI and its context is supplied by `run`.
        let session = unsafe { flint_jpeg_new(write_output) };
        NonNull::new(session)
            .map(Self)
            .ok_or_else(|| Error::Limit("could not allocate JPEG encoder".into()))
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // SAFETY: This is the unique owner. Destroy works after partial creation and
        // does not invoke output callbacks.
        unsafe { flint_jpeg_destroy(self.0) };
    }
}

struct Output<'a> {
    writer: &'a mut Destination,
    failure: Option<WriteFailure>,
}

enum WriteFailure {
    Io(io::Error),
    Panic(Box<dyn Any + Send>),
}

unsafe extern "C" fn write_output(output: *mut c_void, bytes: *const u8, len: usize) -> c_int {
    // SAFETY: The bridge supplies this call's exclusive output context and a
    // readable span of its fixed destination buffer.
    let (output, bytes) = unsafe {
        (
            &mut *output.cast::<Output<'_>>(),
            slice::from_raw_parts(bytes, len),
        )
    };
    output.failure = match catch_unwind(AssertUnwindSafe(|| output.writer.write_all(bytes))) {
        Ok(Ok(())) => return 1,
        Ok(Err(error)) => Some(WriteFailure::Io(error)),
        Err(payload) => Some(WriteFailure::Panic(payload)),
    };
    0
}

unsafe extern "C" {
    fn flint_jpeg_new(
        output: unsafe extern "C" fn(*mut c_void, *const u8, usize) -> c_int,
    ) -> *mut c_void;
    fn flint_jpeg_destroy(session: NonNull<c_void>);
    fn flint_jpeg_error(session: NonNull<c_void>) -> *const c_char;
    fn flint_jpeg_max_dimension() -> u32;
    fn flint_jpeg_start(
        session: NonNull<c_void>,
        output: *mut c_void,
        width: u32,
        height: u32,
        components: c_int,
        quality: c_int,
    ) -> c_int;
    fn flint_jpeg_write_row(session: NonNull<c_void>, output: *mut c_void, row: *const u8)
    -> c_int;
    fn flint_jpeg_finish(session: NonNull<c_void>, output: *mut c_void) -> c_int;
}
