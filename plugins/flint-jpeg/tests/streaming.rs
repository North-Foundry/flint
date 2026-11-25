use std::io::{self, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use flint_jpeg::{JpegEncoder, JpegOptions};
use flint_sdk::{Encoder, Error, ImageInformation, PixelFormat, Rows};

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);

impl Write for Capture {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn start_of_frame(bytes: &[u8]) -> &[u8] {
    assert_eq!(&bytes[..2], &[0xff, 0xd8]);
    assert_eq!(&bytes[bytes.len() - 2..], &[0xff, 0xd9]);
    let mut segments = &bytes[2..];
    loop {
        assert_eq!(segments[0], 0xff);
        let marker = segments[1];
        let length = u16::from_be_bytes([segments[2], segments[3]]) as usize;
        if (0xc0..=0xc3).contains(&marker) {
            assert_eq!(marker, 0xc0, "expected baseline sequential SOF0");
            return &segments[4..2 + length];
        }
        assert_ne!(marker, 0xda, "start of scan without a frame header");
        segments = &segments[2 + length..];
    }
}

#[test]
fn produces_baseline_gray_and_rgb_with_documented_sampling() {
    assert_eq!(JpegOptions::default().quality, 85);
    for (format, component_headers) in [
        (PixelFormat::Gray8, &[1, 0x11, 0][..]),
        (PixelFormat::Rgb8, &[1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1][..]),
    ] {
        let info = ImageInformation::new(3, 2, format).unwrap();
        let output = Capture::default();
        let mut encoder =
            JpegEncoder::open(Box::new(output.clone()), &info, JpegOptions::default()).unwrap();
        let stride = info.row_bytes() + 2;
        let row = vec![42; stride];
        for _ in 0..info.height() {
            encoder
                .write_rows(Rows::new(&info, stride, &row).unwrap())
                .unwrap();
        }
        encoder.finish().unwrap();

        let bytes = output.0.lock().unwrap();
        let frame = start_of_frame(&bytes);
        assert_eq!(frame[..5], [8, 0, 2, 0, 3]);
        assert_eq!(frame[5] as usize, component_headers.len() / 3);
        assert_eq!(&frame[6..], component_headers);
    }
}

#[derive(Clone, Default)]
struct CountBytes(Arc<AtomicUsize>);

impl Write for CountBytes {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.fetch_add(buf.len(), Ordering::Relaxed);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn writes_compressed_bytes_before_receiving_the_full_image() {
    let info = ImageInformation::new(256, 1024, PixelFormat::Rgb8).unwrap();
    let output = CountBytes::default();
    let mut encoder =
        JpegEncoder::open(Box::new(output.clone()), &info, JpegOptions::default()).unwrap();
    let mut row = vec![0; info.row_bytes()];
    let mut state = 1u32;
    for y in 0..info.height() {
        for byte in &mut row {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *byte = state as u8;
        }
        encoder
            .write_rows(Rows::new(&info, row.len(), &row).unwrap())
            .unwrap();
        if y == 127 {
            assert!(output.0.load(Ordering::Relaxed) > 8192);
        }
    }
    encoder.finish().unwrap();
}

struct FailingDestination {
    fail_write: bool,
}

impl Write for FailingDestination {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.fail_write {
            return Err(io::Error::other("write failed"));
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Err(io::Error::other("flush failed"))
    }
}

#[test]
fn reports_destination_errors_and_rejects_invalid_input() {
    let info = ImageInformation::new(1, 2, PixelFormat::Gray8).unwrap();
    let oversized = ImageInformation::new(65_501, 1, PixelFormat::Gray8).unwrap();
    assert!(matches!(
        JpegEncoder::open(Box::new(io::sink()), &oversized, JpegOptions::default()),
        Err(Error::Limit(_))
    ));
    assert!(matches!(
        JpegEncoder::open(Box::new(io::sink()), &info, JpegOptions { quality: 0 }),
        Err(Error::Invalid(_))
    ));

    for (fail_write, message) in [(true, "write failed"), (false, "flush failed")] {
        let destination = Box::new(FailingDestination { fail_write });
        let mut encoder = JpegEncoder::open(destination, &info, JpegOptions::default()).unwrap();
        encoder
            .write_rows(Rows::new(&info, 1, &[42]).unwrap())
            .unwrap();
        assert!(matches!(
            encoder.write_rows(Rows::new(&info, 1, &[42, 43]).unwrap()),
            Err(Error::Invalid(_))
        ));
        encoder
            .write_rows(Rows::new(&info, 1, &[43]).unwrap())
            .unwrap();
        match encoder.finish() {
            Err(Error::Io(error)) => assert_eq!(error.to_string(), message),
            result => panic!("unexpected JPEG encoder result: {result:?}"),
        }
    }
    let incomplete =
        JpegEncoder::open(Box::new(io::sink()), &info, JpegOptions::default()).unwrap();
    assert!(matches!(incomplete.finish(), Err(Error::Invalid(_))));
}

#[cfg(panic = "unwind")]
#[test]
fn resumes_destination_panic_after_returning_from_c() {
    struct Panics;

    impl Write for Panics {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            panic!("destination panicked");
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    let info = ImageInformation::new(1, 1, PixelFormat::Gray8).unwrap();
    let mut encoder = JpegEncoder::open(Box::new(Panics), &info, JpegOptions::default()).unwrap();
    encoder
        .write_rows(Rows::new(&info, 1, &[42]).unwrap())
        .unwrap();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| encoder.finish()));
    let payload = panic.expect_err("expected the destination's panic");
    assert_eq!(
        payload.downcast_ref::<&str>(),
        Some(&"destination panicked")
    );
}
