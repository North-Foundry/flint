use std::io::{self, Read, Write};
use std::sync::{Arc, Mutex};

use flint_png::{PngDecoder, PngEncoder};
use flint_sdk::{Decoder, Encoder, Error, ImageInfo, PixelFormat, Result, Rows, RowsMut, Source};
use png::chunk::{ChunkType, IDAT, IHDR, acTL, eXIf, tRNS};

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

struct ByteAtATime(Vec<u8>);

impl ByteAtATime {
    fn new(mut bytes: Vec<u8>) -> Self {
        bytes.reverse();
        Self(bytes)
    }
}

impl Read for ByteAtATime {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() || self.0.is_empty() {
            return Ok(0);
        }
        buf[0] = self.0.pop().unwrap();
        Ok(1)
    }
}

#[test]
fn round_trips_padded_rows_through_a_forward_only_source() {
    for format in [PixelFormat::Gray8, PixelFormat::Rgb8] {
        let info = ImageInfo::new(2, 2, format).unwrap();
        let packed = info.row_bytes();
        let stride = packed + 2;
        let input: Vec<u8> = (0..(2 * stride) as u8).collect();
        let output = Capture::default();
        let mut encoder = PngEncoder::open(Box::new(output.clone()), &info, ()).unwrap();
        for row in input.chunks_exact(stride) {
            encoder
                .write_rows(Rows::new(&info, stride, row).unwrap())
                .unwrap();
        }
        encoder.finish().unwrap();

        let png = output.0.lock().unwrap();
        assert!(PngDecoder::probe(&png[..8]));
        let source = ByteAtATime::new(png.clone());
        let mut decoder = PngDecoder::open(Box::new(source)).unwrap();
        assert_eq!(*decoder.info(), info);
        let mut buffer = vec![255; 2 * stride];
        for row in buffer.chunks_exact_mut(stride) {
            assert_eq!(
                1,
                decoder
                    .read_rows(RowsMut::new(&info, stride, row).unwrap())
                    .unwrap()
            );
        }
        assert_eq!(
            0,
            decoder
                .read_rows(RowsMut::new(&info, stride, &mut buffer).unwrap())
                .unwrap()
        );
        decoder.finish().unwrap();
        for (actual, expected) in buffer.chunks_exact(stride).zip(input.chunks_exact(stride)) {
            assert_eq!(&actual[..packed], &expected[..packed]);
            assert_eq!(&actual[packed..], &[255, 255]);
        }
    }
}

fn fixture(color: png::ColorType, depth: png::BitDepth, transparent: bool, data: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut encoder = png::Encoder::new(&mut bytes, 1, 1);
    encoder.set_color(color);
    encoder.set_depth(depth);
    if transparent {
        encoder.set_trns(vec![0, 42]);
    }
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(data).unwrap();
    writer.finish().unwrap();
    bytes
}

#[test]
fn verifies_end_chunk_after_all_pixels_are_available() {
    let make_png = || {
        fixture(
            png::ColorType::Grayscale,
            png::BitDepth::Eight,
            false,
            &[42],
        )
    };
    let mut truncated = make_png();
    truncated.truncate(truncated.len() - 12);
    let mut corrupt = make_png();
    *corrupt.last_mut().unwrap() ^= 1;

    for (bytes, missing) in [(truncated, true), (corrupt, false)] {
        let mut decoder = PngDecoder::open(Box::new(io::Cursor::new(bytes))).unwrap();
        let info = *decoder.info();
        let mut pixel = [0];
        assert_eq!(
            1,
            decoder
                .read_rows(RowsMut::new(&info, 1, &mut pixel).unwrap())
                .unwrap()
        );
        assert_eq!(pixel, [42]);
        match decoder.finish() {
            Err(Error::Io(error)) if missing => {
                assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof)
            }
            Err(Error::Codec(_)) if !missing => (),
            result => panic!("unexpected PNG decoder result: {result:?}"),
        }
    }
}

#[test]
fn rejects_transparency_and_high_precision() {
    let gray = png::ColorType::Grayscale;
    let eight = png::BitDepth::Eight;
    for bytes in [
        fixture(gray, eight, true, &[42]),
        fixture(gray, png::BitDepth::Sixteen, false, &[0, 42]),
        fixture(png::ColorType::Rgba, eight, false, &[1, 2, 3, 4]),
    ] {
        assert!(matches!(
            PngDecoder::open(Box::new(io::Cursor::new(bytes))),
            Err(Error::Unsupported(_))
        ));
    }
}

fn encode(info: &ImageInfo, filter: png::Filter, pixels: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut encoder = png::Encoder::new(&mut bytes, info.width(), info.height());
    encoder.set_color(match info.format() {
        PixelFormat::Gray8 => png::ColorType::Grayscale,
        PixelFormat::Rgb8 => png::ColorType::Rgb,
    });
    // Force the requested filter even when filtered data compresses poorly.
    encoder.set_compression(png::Compression::Balanced);
    encoder.set_filter(filter);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(pixels).unwrap();
    writer.finish().unwrap();
    bytes
}

fn decode(source: Source) -> Result<Vec<u8>> {
    let mut decoder = PngDecoder::open(source)?;
    let info = *decoder.info();
    let row_bytes = info.row_bytes();
    let mut pixels = vec![0; row_bytes * info.height() as usize];
    for batch in pixels.chunks_mut(7 * row_bytes) {
        assert_eq!(
            decoder.read_rows(RowsMut::new(&info, row_bytes, batch)?)?,
            batch.len() / row_bytes
        );
    }
    decoder.finish()?;
    Ok(pixels)
}

#[test]
fn matches_reference_for_all_filters_across_inflate_windows() {
    let filters = [
        png::Filter::NoFilter,
        png::Filter::Sub,
        png::Filter::Up,
        png::Filter::Avg,
        png::Filter::Paeth,
    ];
    for format in [PixelFormat::Gray8, PixelFormat::Rgb8] {
        // Vary both neighbors and generate repeated segments across inflate windows.
        let info = ImageInfo::new(701, 117, format).unwrap();
        let pixels: Vec<_> = (0..info.height() as usize)
            .flat_map(|y| {
                (0..info.row_bytes()).map(move |x| (17 * x + (y % 13) * (3 + (x % 29))) as u8)
            })
            .collect();
        for filter in filters {
            let bytes = encode(&info, filter, &pixels);
            let mut reference = png::Decoder::new(io::Cursor::new(&bytes))
                .read_info()
                .unwrap();
            let mut expected = vec![0; reference.output_buffer_size().unwrap()];
            reference.next_frame(&mut expected).unwrap();
            assert_eq!(pixels, expected);
            assert_eq!(
                decode(Box::new(io::Cursor::new(bytes))).unwrap(),
                expected,
                "{format:?} {filter:?}"
            );
        }
    }
}

fn compressed_payload(bytes: &[u8]) -> Vec<u8> {
    let mut chunks = &bytes[8..];
    let mut payload = Vec::new();
    while !chunks.is_empty() {
        let length = u32::from_be_bytes(chunks[..4].try_into().unwrap()) as usize;
        if &chunks[4..8] == b"IDAT" {
            payload.extend_from_slice(&chunks[8..8 + length]);
        }
        chunks = &chunks[length + 12..];
    }
    payload
}

fn png_with_chunks(width: u32, height: u32, chunks: &[(ChunkType, &[u8])]) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut writer = png::Encoder::new(&mut bytes, width, height)
        .write_header()
        .unwrap();
    for &(kind, data) in chunks {
        writer.write_chunk(kind, data).unwrap();
    }
    writer.finish().unwrap();
    bytes
}

#[test]
fn verifies_split_zlib_trailers_after_delivering_pixels() {
    // The filtered image exactly fills the larger inflate window. The decoder
    // must compact it before validating the trailer with spare output space.
    let info = ImageInfo::new(32_896, 3, PixelFormat::Gray8).unwrap();
    let expected = vec![42; info.row_bytes() * info.height() as usize];
    let payload = compressed_payload(&encode(&info, png::Filter::Paeth, &expected));
    let (deflate, adler) = payload.split_at(payload.len() - 4);
    for (tail, valid) in [
        (adler.to_vec(), true),
        (vec![], false),
        (adler[..2].to_vec(), false),
        (vec![adler[0], adler[1], adler[2] ^ 1, adler[3]], false),
    ] {
        let mut chunks = vec![(IDAT, deflate), (IDAT, &[][..])];
        chunks.extend(tail.chunks(1).map(|byte| (IDAT, byte)));
        let bytes = png_with_chunks(info.width(), info.height(), &chunks);
        let mut decoder = PngDecoder::open(Box::new(ByteAtATime::new(bytes))).unwrap();
        let mut pixels = vec![0; expected.len()];
        let output = RowsMut::new(&info, info.row_bytes(), &mut pixels).unwrap();
        assert_eq!(decoder.read_rows(output).unwrap(), info.height() as usize);
        assert_eq!(pixels, expected);
        let result = decoder.finish();
        if valid {
            result.unwrap();
        } else {
            assert!(matches!(result, Err(Error::Codec(_))), "{result:?}");
        }
    }
}

#[test]
fn checksums_queued_matches_across_inflate_window_compaction() {
    // This RGB stream needs to drain a queued DEFLATE match near a window end.
    // Giving fdeflate 0.3.7 just one spare byte can produce WrongChecksum here.
    let info = ImageInfo::new(2048, 512, PixelFormat::Rgb8).unwrap();
    let expected: Vec<_> = (0..info.height() as usize)
        .flat_map(|y| (0..info.row_bytes()).map(move |x| (17 * x + (y % 33) * (3 + x % 29)) as u8))
        .collect();
    let bytes = encode(&info, png::Filter::Avg, &expected);
    assert_eq!(decode(Box::new(io::Cursor::new(bytes))).unwrap(), expected);
}

#[test]
fn validates_the_decompressed_size_and_idat_completion() {
    let info = ImageInfo::new(2, 1, PixelFormat::Gray8).unwrap();
    let payload = compressed_payload(&encode(&info, png::Filter::NoFilter, &[42, 43]));
    for width in [1, 3] {
        let bytes = png_with_chunks(width, 1, &[(IDAT, &payload)]);
        let result = decode(Box::new(io::Cursor::new(bytes)));
        match (width, result) {
            (1, Err(Error::Codec(message))) => assert!(message.contains("excess decompressed")),
            (3, Err(Error::Io(error))) => assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof),
            (width, result) => panic!("unexpected result for width {width}: {result:?}"),
        }
    }

    let mut bytes = png_with_chunks(2, 1, &[(IDAT, &payload)]);
    // Cut the file inside IDAT's CRC, with the zlib stream itself still valid.
    bytes.truncate(bytes.len() - 13);
    let result = decode(Box::new(ByteAtATime::new(bytes)));
    assert!(
        matches!(result, Err(Error::Io(error)) if error.kind() == io::ErrorKind::UnexpectedEof)
    );
}

#[test]
fn bounds_png_metadata_and_row_allocations() {
    let info = ImageInfo::new(1, 1, PixelFormat::Gray8).unwrap();
    let fixture = encode(&info, png::Filter::NoFilter, &[42]);
    let payload = compressed_payload(&fixture);
    let unknown = ChunkType(*b"vpAg");
    let exif_cap = 1024 * 1024;
    for (kind, length) in [
        (eXIf, exif_cap + 1),
        (unknown, 1u32 << 31),
        (IDAT, 1u32 << 31),
    ] {
        let mut header = fixture[..33].to_vec();
        header.extend_from_slice(&length.to_be_bytes());
        header.extend_from_slice(&kind.0);
        assert!(matches!(
            PngDecoder::open(Box::new(io::Cursor::new(header))),
            Err(Error::Limit(_))
        ));
    }

    let skipped = vec![0; exif_cap as usize + 1];
    let bytes = png_with_chunks(1, 1, &[(unknown, &skipped), (IDAT, &payload)]);
    assert_eq!(decode(Box::new(io::Cursor::new(bytes))).unwrap(), [42]);
    let bytes = png_with_chunks(1, 1, &[(IDAT, &payload), (eXIf, &skipped)]);
    assert!(matches!(
        decode(Box::new(io::Cursor::new(bytes))),
        Err(Error::Limit(_))
    ));

    let bytes = png_with_chunks(16 * 1024 * 1024 + 1, 1, &[(IDAT, &payload)]);
    assert!(matches!(
        PngDecoder::open(Box::new(io::Cursor::new(bytes))),
        Err(Error::Limit(_))
    ));

    let bytes = png_with_chunks(1, 1, &[(acTL, &[0, 0, 0, 1, 0, 0, 0, 0]), (IDAT, &payload)]);
    assert!(matches!(
        PngDecoder::open(Box::new(io::Cursor::new(bytes))),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn rejects_zero_dimensions_and_transparency_on_either_side_of_idat() {
    for (width, height) in [(0u32, 1u32), (1, 0)] {
        let mut header = [0; 13];
        header[..4].copy_from_slice(&width.to_be_bytes());
        header[4..8].copy_from_slice(&height.to_be_bytes());
        header[8] = 8;
        let mut bytes = png_with_chunks(1, 1, &[(IHDR, &header)]);
        bytes.drain(8..33); // Leave the synthetic IHDR as the first chunk.
        assert!(matches!(
            PngDecoder::open(Box::new(io::Cursor::new(bytes))),
            Err(Error::Codec(message)) if message.to_lowercase().contains("dimension")
        ));
    }

    let info = ImageInfo::new(1, 1, PixelFormat::Gray8).unwrap();
    let payload = compressed_payload(&encode(&info, png::Filter::NoFilter, &[42]));
    let image = (IDAT, payload.as_slice());
    let transparency = (tRNS, &[0, 42][..]);
    let bytes = png_with_chunks(1, 1, &[transparency, image]);
    assert!(matches!(
        PngDecoder::open(Box::new(io::Cursor::new(bytes))),
        Err(Error::Unsupported(message)) if message.contains("tRNS")
    ));

    let bytes = png_with_chunks(1, 1, &[image, transparency]);
    let mut decoder = PngDecoder::open(Box::new(ByteAtATime::new(bytes))).unwrap();
    let mut pixel = [0];
    assert_eq!(
        decoder
            .read_rows(RowsMut::new(&info, 1, &mut pixel).unwrap())
            .unwrap(),
        1
    );
    assert!(
        matches!(decoder.finish(), Err(Error::Unsupported(message)) if message.contains("tRNS"))
    );
}

struct FailingDestination {
    fail_at_iend: bool,
}

impl Write for FailingDestination {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.fail_at_iend && buf == b"IEND" {
            return Err(io::Error::other("IEND failed"));
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Err(io::Error::other("flush failed"))
    }
}

#[test]
fn reports_iend_and_flush_errors_and_rejects_missing_rows() {
    let info = ImageInfo::new(1, 1, PixelFormat::Gray8).unwrap();
    for (fail_at_iend, message) in [(true, "IEND failed"), (false, "flush failed")] {
        let dest = Box::new(FailingDestination { fail_at_iend });
        let mut encoder = PngEncoder::open(dest, &info, ()).unwrap();
        encoder
            .write_rows(Rows::new(&info, 1, &[42]).unwrap())
            .unwrap();
        match encoder.finish() {
            Err(Error::Io(error)) => assert_eq!(error.to_string(), message),
            result => panic!("unexpected PNG encoder result: {result:?}"),
        }
    }
    let incomplete = PngEncoder::open(Box::new(io::sink()), &info, ()).unwrap();
    assert!(matches!(incomplete.finish(), Err(Error::Invalid(_))));
}
