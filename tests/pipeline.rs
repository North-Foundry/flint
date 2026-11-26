use std::io::{Cursor, Read, Write};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

#[cfg(all(feature = "png", feature = "jpeg"))]
use flint::Image;
use flint::sdk::{
    Decoder, Destination, Encoder, Expr, ImageExpression, ImageInformation, Operation, Pixel,
    PixelFormat, Rows, RowsMut, Source,
};
use flint::{Backend, Engine, Error, Optimization, Result};
use flint_imaging::{
    blur::GaussianBlur,
    color::{Brightness, Grayscale},
};

const MAGIC: &[u8; 4] = b"FRAW";

fn raw_image(width: u32, height: u32, pixels: &[u8]) -> Vec<u8> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|size| size.checked_mul(PixelFormat::Rgb8.channels()))
        .expect("fixture dimensions fit address space");
    assert_eq!(pixels.len(), expected);
    let mut bytes = MAGIC.to_vec();
    bytes.extend_from_slice(&width.to_le_bytes());
    bytes.extend_from_slice(&height.to_le_bytes());
    bytes.extend_from_slice(pixels);
    bytes
}

struct RawDecoder {
    info: ImageInformation,
    source: Source,
    rows_read: u32,
}

impl Decoder for RawDecoder {
    const NAME: &'static str = "test raw RGB";
    const EXTENSIONS: &'static [&'static str] = &["fraw"];

    fn probe(header: &[u8]) -> bool {
        header.starts_with(MAGIC)
    }

    fn open(mut source: Source) -> Result<Self> {
        let mut header = [0; 12];
        source.read_exact(&mut header)?;
        if &header[..4] != MAGIC {
            return Err(Error::Invalid("bad raw image signature".into()));
        }
        let width = u32::from_le_bytes(header[4..8].try_into().unwrap());
        let height = u32::from_le_bytes(header[8..12].try_into().unwrap());
        Ok(Self {
            info: ImageInformation::new(width, height, PixelFormat::Rgb8)?,
            source,
            rows_read: 0,
        })
    }

    fn info(&self) -> &ImageInformation {
        &self.info
    }

    fn read_rows(&mut self, mut output: RowsMut<'_>) -> Result<usize> {
        let count = output
            .len()
            .min((self.info.height() - self.rows_read) as usize);
        for row in 0..count {
            self.source
                .read_exact(output.row_mut(row))
                .map_err(|error| {
                    if error.kind() == std::io::ErrorKind::UnexpectedEof {
                        Error::Codec("truncated raw image pixels".into())
                    } else {
                        error.into()
                    }
                })?;
        }
        self.rows_read += count as u32;
        Ok(count)
    }
}

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);

impl Capture {
    fn bytes(&self) -> Vec<u8> {
        self.0.lock().unwrap().clone()
    }
}

struct SharedWriter(Arc<Mutex<Vec<u8>>>);

impl Write for SharedWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[derive(Default)]
struct SinkStats {
    calls: AtomicUsize,
    bytes: AtomicUsize,
}

struct StatsWriter(Arc<SinkStats>);

impl Write for StatsWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.calls.fetch_add(1, Ordering::Relaxed);
        self.0.bytes.fetch_add(bytes.len(), Ordering::Relaxed);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

struct RawEncoder {
    destination: Destination,
}

static COUNTING_DECODER_OPENS: AtomicUsize = AtomicUsize::new(0);
static COUNTING_ENCODER_OPENS: AtomicUsize = AtomicUsize::new(0);

struct CountingDecoder(RawDecoder);

impl Decoder for CountingDecoder {
    const NAME: &'static str = "counting test raw RGB";
    const EXTENSIONS: &'static [&'static str] = &["counting-raw"];

    fn probe(header: &[u8]) -> bool {
        RawDecoder::probe(header)
    }

    fn open(source: Source) -> Result<Self> {
        COUNTING_DECODER_OPENS.fetch_add(1, Ordering::SeqCst);
        Ok(Self(RawDecoder::open(source)?))
    }

    fn info(&self) -> &ImageInformation {
        self.0.info()
    }

    fn read_rows(&mut self, rows: RowsMut<'_>) -> Result<usize> {
        self.0.read_rows(rows)
    }

    fn finish(self) -> Result<()> {
        self.0.finish()
    }
}

struct CountingEncoder(RawEncoder);

impl Encoder for CountingEncoder {
    const NAME: &'static str = "counting test raw RGB encoder";
    const EXTENSIONS: &'static [&'static str] = &["counting-raw"];
    type Options = ();

    fn open(destination: Destination, info: &ImageInformation, options: ()) -> Result<Self> {
        COUNTING_ENCODER_OPENS.fetch_add(1, Ordering::SeqCst);
        Ok(Self(RawEncoder::open(destination, info, options)?))
    }

    fn write_rows(&mut self, rows: Rows<'_>) -> Result<()> {
        self.0.write_rows(rows)
    }

    fn finish(self) -> Result<()> {
        self.0.finish()
    }
}

impl Encoder for RawEncoder {
    const NAME: &'static str = "test raw RGB encoder";
    const EXTENSIONS: &'static [&'static str] = &["fraw"];
    type Options = ();

    fn open(mut destination: Destination, info: &ImageInformation, _: ()) -> Result<Self> {
        destination.write_all(MAGIC)?;
        destination.write_all(&info.width().to_le_bytes())?;
        destination.write_all(&info.height().to_le_bytes())?;
        Ok(Self { destination })
    }

    fn write_rows(&mut self, rows: Rows<'_>) -> Result<()> {
        for row in 0..rows.len() {
            self.destination.write_all(rows.row(row))?;
        }
        self.destination.flush()?;
        Ok(())
    }

    fn finish(self) -> Result<()> {
        Ok(())
    }
}

fn engine() -> Result<Engine> {
    Engine::builder()
        .register_decoder::<RawDecoder>()
        .register_encoder::<RawEncoder>()
        .build()
}

fn raw_engine(backend: Backend) -> Result<Engine> {
    Engine::builder()
        .backend(backend)
        .register_decoder::<RawDecoder>()
        .register_encoder::<RawEncoder>()
        .build()
}

fn limited_raw_engine(backend: Backend, memory_limit: usize) -> Result<Engine> {
    Engine::builder()
        .backend(backend)
        .memory_limit(memory_limit)
        .register_decoder::<RawDecoder>()
        .register_encoder::<RawEncoder>()
        .build()
}

/**
 * Build a raw-codec engine with explicit backend, optimization, and memory settings.
 */
fn configured_raw_engine(
    backend: Backend,
    optimization: Optimization,
    memory_limit: usize,
) -> Result<Engine> {
    Engine::builder()
        .backend(backend)
        .optimization(optimization)
        .memory_limit(memory_limit)
        .register_decoder::<RawDecoder>()
        .register_encoder::<RawEncoder>()
        .build()
}

struct CountingReader {
    inner: Cursor<Vec<u8>>,
    bytes_read: Arc<AtomicUsize>,
    max_chunk: usize,
}

impl Read for CountingReader {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        let limit = output.len().min(self.max_chunk);
        let count = self.inner.read(&mut output[..limit])?;
        self.bytes_read.fetch_add(count, Ordering::SeqCst);
        Ok(count)
    }
}

fn source(
    bytes: Vec<u8>,
    bytes_read: Arc<AtomicUsize>,
) -> impl Fn() -> Result<Source> + Send + Sync {
    move || {
        Ok(Box::new(CountingReader {
            inner: Cursor::new(bytes.clone()),
            bytes_read: bytes_read.clone(),
            max_chunk: 5,
        }))
    }
}

struct GeneratedReader {
    header: [u8; 12],
    position: u128,
    total: u128,
}

impl GeneratedReader {
    fn new(width: u32, height: u32) -> Self {
        let mut header = [0; 12];
        header[..4].copy_from_slice(MAGIC);
        header[4..8].copy_from_slice(&width.to_le_bytes());
        header[8..12].copy_from_slice(&height.to_le_bytes());
        Self {
            header,
            position: 0,
            total: 12 + u128::from(width) * u128::from(height) * 3,
        }
    }
}

impl Read for GeneratedReader {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        let remaining = self.total.saturating_sub(self.position);
        let count = output
            .len()
            .min(usize::try_from(remaining).unwrap_or(usize::MAX));
        for (index, byte) in output[..count].iter_mut().enumerate() {
            let offset = self.position + index as u128;
            *byte = if offset < 12 {
                self.header[offset as usize]
            } else {
                let pixel_index = offset - 12;
                ((pixel_index * 73 + pixel_index / 11 * 19 + 31) % 256) as u8
            };
        }
        self.position += count as u128;
        Ok(count)
    }
}

fn generated_source(width: u32, height: u32) -> impl Fn() -> Result<Source> + Send + Sync {
    move || Ok(Box::new(GeneratedReader::new(width, height)))
}

fn render<O: flint::sdk::Operation>(
    pixels: &[u8],
    width: u32,
    height: u32,
    operation: O,
    backend: Backend,
) -> Result<Vec<u8>> {
    let capture = Capture::default();
    let engine = raw_engine(backend)?;
    let image = engine.open_source(source(
        raw_image(width, height, pixels),
        Arc::new(AtomicUsize::new(0)),
    ))?;
    image.apply(operation)?.write(
        "fraw",
        Box::new(SharedWriter(capture.0.clone())) as Destination,
    )?;
    Ok(capture.bytes()[12..].to_vec())
}

/**
 * Render an operation through a public engine configured for one optimization mode.
 */
fn render_configured<O: Operation>(
    pixels: &[u8],
    width: u32,
    height: u32,
    operation: O,
    backend: Backend,
    optimization: Optimization,
) -> Result<Vec<u8>> {
    let capture = Capture::default();
    let engine = configured_raw_engine(backend, optimization, 1_000_000)?;
    engine
        .open_source(source(
            raw_image(width, height, pixels),
            Arc::new(AtomicUsize::new(0)),
        ))?
        .apply(operation)?
        .write(
            "fraw",
            Box::new(SharedWriter(capture.0.clone())) as Destination,
        )?;
    Ok(capture.bytes()[12..].to_vec())
}

/**
 * Assert byte equality between automatic and disabled optimization for one graph.
 */
fn assert_optimization_matches<O: Operation>(
    pixels: &[u8],
    width: u32,
    height: u32,
    backend: Backend,
    operation: impl Fn() -> O,
) -> Result<()> {
    assert_eq!(
        render_configured(
            pixels,
            width,
            height,
            operation(),
            backend,
            Optimization::Auto,
        )?,
        render_configured(
            pixels,
            width,
            height,
            operation(),
            backend,
            Optimization::Disabled,
        )?,
    );
    Ok(())
}

fn reference_brightness(pixels: &[u8], amount: f32) -> Vec<u8> {
    pixels
        .iter()
        .map(|&sample| quantize(sample as f32 / 255.0 + amount))
        .collect()
}

fn quantize(value: f32) -> u8 {
    let value = if value.is_finite() { value } else { 0.0 };
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn reference_gray(rgb: &[u8]) -> Vec<u8> {
    rgb.as_chunks::<3>()
        .0
        .iter()
        .map(|p| {
            quantize(
                p[0] as f32 / 255.0 * 0.2126
                    + p[1] as f32 / 255.0 * 0.7152
                    + p[2] as f32 / 255.0 * 0.0722,
            )
        })
        .collect()
}

fn reference_gaussian(
    rgb: &[u8],
    width: usize,
    height: usize,
    sigma: f32,
    quantize_between_passes: bool,
) -> Vec<u8> {
    let radius = (3.0 * sigma).ceil() as i32;
    let mut weights: Vec<f32> = (-radius..=radius)
        .map(|offset| {
            (-(f64::from(offset).powi(2)) / (2.0 * f64::from(sigma).powi(2))).exp() as f32
        })
        .collect();
    let sum: f32 = weights.iter().sum();
    weights.iter_mut().for_each(|weight| *weight /= sum);

    let sample =
        |pixels: &[f32], x: usize, y: usize, channel: usize| pixels[(y * width + x) * 3 + channel];
    let source: Vec<f32> = rgb.iter().map(|&value| f32::from(value) / 255.0).collect();
    let mut horizontal = vec![0.0; rgb.len()];
    for y in 0..height {
        for x in 0..width {
            for channel in 0..3 {
                let mut value = 0.0_f32;
                for offset in -radius..=radius {
                    let sample_x = x.saturating_add_signed(offset as isize).min(width - 1);
                    value +=
                        sample(&source, sample_x, y, channel) * weights[(offset + radius) as usize];
                }
                horizontal[(y * width + x) * 3 + channel] = if quantize_between_passes {
                    f32::from(quantize(value)) / 255.0
                } else {
                    value
                };
            }
        }
    }

    let mut result = vec![0; rgb.len()];
    for y in 0..height {
        for x in 0..width {
            for channel in 0..3 {
                let mut value = 0.0_f32;
                for offset in -radius..=radius {
                    let sample_y = y.saturating_add_signed(offset as isize).min(height - 1);
                    value += sample(&horizontal, x, sample_y, channel)
                        * weights[(offset + radius) as usize];
                }
                result[(y * width + x) * 3 + channel] = quantize(value);
            }
        }
    }
    result
}

struct ShiftLeftThenDown;

impl Operation for ShiftLeftThenDown {
    fn apply(&self, image: ImageExpression) -> Result<ImageExpression> {
        image.stencil(|n| n.at(-1, 0))?.stencil(|n| n.at(0, 1))
    }
}

#[test]
fn codec_registration_is_lazy_and_apply_does_not_decode() -> Result<()> {
    let decoder_before = COUNTING_DECODER_OPENS.load(Ordering::SeqCst);
    let encoder_before = COUNTING_ENCODER_OPENS.load(Ordering::SeqCst);
    let factory_calls = Arc::new(AtomicUsize::new(0));
    let bytes_read = Arc::new(AtomicUsize::new(0));
    let input = raw_image(3, 2, &[10; 18]);
    let factory_calls_for_source = factory_calls.clone();
    let source_bytes_read = bytes_read.clone();
    let engine = Engine::builder()
        .register_decoder::<CountingDecoder>()
        .register_encoder::<CountingEncoder>()
        .build()?;
    assert_eq!(
        COUNTING_DECODER_OPENS.load(Ordering::SeqCst),
        decoder_before
    );
    assert_eq!(
        COUNTING_ENCODER_OPENS.load(Ordering::SeqCst),
        encoder_before
    );
    let image = engine.open_source(move || {
        factory_calls_for_source.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(CountingReader {
            inner: Cursor::new(input.clone()),
            bytes_read: source_bytes_read.clone(),
            max_chunk: 5,
        }))
    })?;
    assert_eq!(
        COUNTING_DECODER_OPENS.load(Ordering::SeqCst),
        decoder_before + 1
    );
    assert_eq!(factory_calls.load(Ordering::SeqCst), 1);
    let read_after_open = bytes_read.load(Ordering::SeqCst);
    let processed = image.apply(Brightness::new(0.1))?;
    assert_eq!(
        bytes_read.load(Ordering::SeqCst),
        read_after_open,
        "apply must not decode pixels"
    );
    assert_eq!(processed.metadata().width(), 3);
    assert_eq!(
        COUNTING_ENCODER_OPENS.load(Ordering::SeqCst),
        encoder_before
    );
    let sink_stats = Arc::new(SinkStats::default());
    processed.write(
        "counting-raw",
        Box::new(StatsWriter(sink_stats)) as Destination,
    )?;
    assert_eq!(
        COUNTING_DECODER_OPENS.load(Ordering::SeqCst),
        decoder_before + 2
    );
    assert_eq!(
        COUNTING_ENCODER_OPENS.load(Ordering::SeqCst),
        encoder_before + 1
    );
    assert_eq!(factory_calls.load(Ordering::SeqCst), 2);
    Ok(())
}

#[test]
fn brightness_matches_independent_reference() -> Result<()> {
    let input = [0, 12, 64, 128, 200, 255];
    assert_eq!(
        render(&input, 2, 1, Brightness::new(0.125), Backend::Scalar)?,
        reference_brightness(&input, 0.125)
    );
    Ok(())
}

#[test]
fn grayscale_reference_uses_encoded_space_luma() {
    assert_eq!(reference_gray(&[255, 0, 0]), [54]);
    assert_eq!(reference_gray(&[0, 255, 0]), [182]);
    assert_eq!(reference_gray(&[0, 0, 255]), [18]);
}

#[test]
fn grayscale_and_simd_match_scalar_reference() -> Result<()> {
    // Seven pixels exercise a vector tail on common SIMD widths.
    let input: Vec<u8> = (0..21).map(|i| (i * 37) as u8).collect();
    let expected = reference_gray(&input);
    assert_eq!(render(&input, 7, 1, Grayscale, Backend::Scalar)?, expected);
    let simd = render(&input, 7, 1, Grayscale, Backend::Simd)?;
    assert_eq!(simd, expected);
    Ok(())
}

#[test]
fn arithmetic_semantics_match_in_vectors_and_scalar_tails() -> Result<()> {
    #[derive(Clone, Copy)]
    struct Arithmetic;

    impl Operation for Arithmetic {
        fn apply(&self, image: ImageExpression) -> Result<ImageExpression> {
            image.map_pixel(|p| {
                Pixel::rgb(
                    ((p.r - 0.25) / 0.5).clamp(0.0, 1.0),
                    1.0 / (p.g * -0.0).min(0.0),
                    (p.b / 0.0).min(f32::from_bits(0x7f80_0001)),
                )
            })
        }
    }

    let rgb = [
        [0, 20, 0],
        [255, 50, 1],
        [127, 40, 0],
        [1, 70, 2],
        [200, 25, 3],
        [10, 55, 0],
        [252, 0, 0],
    ];
    let input = rgb.as_flattened();
    let expected: Vec<_> = rgb
        .into_iter()
        .flat_map(|[r, _, b]| {
            [
                quantize((f32::from(r) / 255.0 - 0.25) / 0.5),
                255,
                if b == 0 { 0 } else { 255 },
            ]
        })
        .collect();
    assert_eq!(render(input, 7, 1, Arithmetic, Backend::Scalar)?, expected);
    assert_eq!(render(input, 7, 1, Arithmetic, Backend::Simd)?, expected);
    Ok(())
}

#[test]
fn gaussian_rejects_invalid_or_oversized_sigma() -> Result<()> {
    let image = raw_engine(Backend::Scalar)?.open_source(generated_source(4, 4))?;
    for sigma in [0.0, -1.0, f32::INFINITY, f32::NAN] {
        assert!(matches!(
            image.clone().apply(GaussianBlur::new(sigma)),
            Err(Error::Invalid(_))
        ));
    }
    assert!(matches!(
        image.apply(GaussianBlur::new(f32::MAX)),
        Err(Error::Limit(_))
    ));
    Ok(())
}

#[test]
fn gaussian_matches_unquantized_two_pass_reference() -> Result<()> {
    let width = 9;
    let height = 7;
    let input: Vec<u8> = (0..width * height * 3)
        .map(|i| ((i * 73 + i / 11 * 19 + 31) % 256) as u8)
        .collect();
    let expected = reference_gaussian(&input, width, height, 1.0, false);
    let quantized_between_passes = reference_gaussian(&input, width, height, 1.0, true);
    assert_ne!(expected, quantized_between_passes);
    assert_eq!(
        render(
            &input,
            width as u32,
            height as u32,
            GaussianBlur::new(1.0),
            Backend::Scalar
        )?,
        expected
    );
    Ok(())
}

#[test]
fn signed_offsets_borders_and_consecutive_stages_match_reference() -> Result<()> {
    let (width, height) = (4_usize, 3_usize);
    let input: Vec<u8> = (0..height)
        .flat_map(|y| {
            (0..width).flat_map(move |x| {
                let value = (10 + y * 50 + x * 7) as u8;
                [value, value + 1, value + 2]
            })
        })
        .collect();
    let mut expected = Vec::with_capacity(input.len());
    for y in 0..height {
        for x in 0..width {
            let source_x = x.saturating_sub(1);
            let source_y = (y + 1).min(height - 1);
            let pixel = (source_y * width + source_x) * 3;
            expected.extend_from_slice(&input[pixel..pixel + 3]);
        }
    }
    assert_eq!(
        render(
            &input,
            width as u32,
            height as u32,
            ShiftLeftThenDown,
            Backend::Scalar
        )?,
        expected
    );

    let plan = raw_engine(Backend::Scalar)?
        .open_source(generated_source(width as u32, height as u32))?
        .apply(ShiftLeftThenDown)?
        .explain()?;
    assert_eq!((plan.stages[0].min_x, plan.stages[0].max_x), (-1, -1));
    assert_eq!((plan.stages[0].min_y, plan.stages[0].max_y), (0, 0));
    assert_eq!((plan.stages[1].min_y, plan.stages[1].max_y), (1, 1));
    Ok(())
}

#[test]
fn stencil_plan_reports_spatial_dependencies_and_bounded_rows() -> Result<()> {
    let small = raw_engine(Backend::Scalar)?
        .open_source(generated_source(8, 24))?
        .apply(GaussianBlur::new(1.0))?;
    let large = raw_engine(Backend::Scalar)?
        .open_source(generated_source(8, 1200))?
        .apply(GaussianBlur::new(1.0))?;
    let small_plan = small.explain()?;
    let large_plan = large.explain()?;
    assert_eq!(small_plan.backend, Backend::Scalar);
    assert_eq!(large_plan.backend, Backend::Scalar);
    assert!(small_plan.stages.iter().any(|s| s.min_y < 0 || s.max_y > 0));
    assert!(
        large_plan
            .stages
            .iter()
            .map(|s| s.buffered_rows)
            .max()
            .unwrap_or(0)
            < 1200
    );
    assert_eq!(small_plan.managed_bytes, large_plan.managed_bytes);
    Ok(())
}

#[test]
fn generated_large_image_streams_to_sink_with_height_independent_plan() -> Result<()> {
    let (width, short_height, tall_height) = (9, 64, 10_000);
    let memory_limit = 1_000_000;
    let short = limited_raw_engine(Backend::Scalar, memory_limit)?
        .open_source(generated_source(width, short_height))?
        .apply(GaussianBlur::new(1.0))?;
    let tall = limited_raw_engine(Backend::Scalar, memory_limit)?
        .open_source(generated_source(width, tall_height))?
        .apply(GaussianBlur::new(1.0))?;
    let short_plan = short.explain()?;
    let tall_plan = tall.explain()?;
    assert_eq!(short_plan.managed_bytes, tall_plan.managed_bytes);
    assert!(short_plan.managed_bytes <= memory_limit);

    let short_sink = Arc::new(SinkStats::default());
    let short_stats = short.write_with_statistics(
        "fraw",
        Box::new(StatsWriter(short_sink.clone())) as Destination,
    )?;
    let tall_sink = Arc::new(SinkStats::default());
    let tall_stats = tall.write_with_statistics(
        "fraw",
        Box::new(StatsWriter(tall_sink.clone())) as Destination,
    )?;
    assert_eq!(short_stats.backend, Backend::Scalar);
    assert_eq!(tall_stats.backend, Backend::Scalar);
    assert!(short_stats.peak_managed_bytes > 0);
    assert!(tall_stats.peak_managed_bytes > 0);
    assert!(short_stats.peak_managed_bytes >= short_plan.managed_bytes);
    assert!(tall_stats.peak_managed_bytes >= tall_plan.managed_bytes);
    assert!(short_stats.peak_managed_bytes <= memory_limit);
    assert!(tall_stats.peak_managed_bytes <= memory_limit);
    assert_eq!(
        short_stats.peak_managed_bytes,
        tall_stats.peak_managed_bytes
    );
    assert!(short_sink.calls.load(Ordering::Relaxed) >= short_height as usize + 3);
    assert!(tall_sink.calls.load(Ordering::Relaxed) >= tall_height as usize + 3);
    assert_eq!(
        short_sink.bytes.load(Ordering::Relaxed),
        12 + width as usize * short_height as usize * 3
    );
    assert_eq!(
        tall_sink.bytes.load(Ordering::Relaxed),
        12 + width as usize * tall_height as usize * 3
    );
    Ok(())
}

#[test]
fn gaussian_blur_preserves_constant_image_even_when_image_is_smaller_than_radius() -> Result<()> {
    let input = [73; 3 * 2 * 3];
    let output = render(&input, 3, 2, GaussianBlur::new(2.0), Backend::Scalar)?;
    assert_eq!(output, input);
    Ok(())
}

#[test]
fn memory_budget_and_corrupt_input_return_errors() -> Result<()> {
    let limited = limited_raw_engine(Backend::Scalar, 1)?;
    let pixels = vec![120; 4 * 4 * 3];
    let image = limited.open_source(source(
        raw_image(4, 4, &pixels),
        Arc::new(AtomicUsize::new(0)),
    ))?;
    let capture = Capture::default();
    let result = image.write(
        "fraw",
        Box::new(SharedWriter(capture.0.clone())) as Destination,
    );
    assert!(matches!(result, Err(Error::Limit(_))));

    let engine = engine()?;
    let corrupt = engine.open_source(source(
        b"not a supported image".to_vec(),
        Arc::new(AtomicUsize::new(0)),
    ));
    assert!(corrupt.is_err());

    let mut invalid_dimensions = MAGIC.to_vec();
    invalid_dimensions.extend_from_slice(&0_u32.to_le_bytes());
    invalid_dimensions.extend_from_slice(&1_u32.to_le_bytes());
    assert!(
        engine
            .open_source(source(invalid_dimensions, Arc::new(AtomicUsize::new(0)),))
            .is_err()
    );

    // Metadata is cheap to inspect, but this enormous row must fail the managed
    // allocation budget before execution attempts to allocate pixel buffers.
    let huge_engine = limited_raw_engine(Backend::Scalar, 1024)?;
    match huge_engine.open_source(generated_source(u32::MAX, u32::MAX)) {
        Err(Error::Limit(_)) => (), // The packed RGB row is unaddressable on 32-bit targets.
        Ok(image) => {
            let huge = image.apply(GaussianBlur::new(1.0))?;
            let plan = huge.explain()?;
            assert!(plan.managed_bytes > 1024);
            assert!(matches!(
                huge.write("fraw", Box::new(std::io::sink())),
                Err(Error::Limit(_))
            ));
        }
        Err(error) => return Err(error),
    }
    Ok(())
}

#[test]
fn save_keeps_existing_destination_when_stream_is_truncated() -> Result<()> {
    let original = b"preserve the old output";
    let full = raw_image(3, 2, &[80; 18]);
    let mut truncated = full.clone();
    truncated.truncate(12 + 3 * 3);
    let opens = Arc::new(AtomicUsize::new(0));
    let opens_for_source = opens.clone();
    let image = engine()?.open_source(move || {
        let bytes = if opens_for_source.fetch_add(1, Ordering::SeqCst) == 0 {
            full.clone()
        } else {
            truncated.clone()
        };
        Ok(Box::new(Cursor::new(bytes)))
    })?;

    let dir = tempfile::tempdir()?;
    let path = dir.path().join("existing.fraw");
    std::fs::write(&path, original)?;
    let result = image.apply(Brightness::new(0.1))?.save(&path);
    assert!(matches!(result, Err(Error::Codec(_))));
    assert_eq!(std::fs::read(&path)?, original);
    Ok(())
}

#[cfg(all(feature = "png", feature = "jpeg"))]
#[test]
fn png_to_jpeg_file_conversion() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let input = dir.path().join("input.data");
    let output = dir.path().join("output.JpG");
    let file = std::fs::File::create(&input)?;
    let mut encoder = png::Encoder::new(file, 2, 2);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|e| Error::Codec(e.to_string()))?;
    writer
        .write_image_data(&[255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])
        .map_err(|e| Error::Codec(e.to_string()))?;
    drop(writer);

    Image::open(&input)?.save(&output)?;
    let encoded = std::fs::read(&output)?;
    assert!(
        encoded.starts_with(&[0xff, 0xd8]),
        "expected JPEG SOI marker"
    );
    assert!(encoded.ends_with(&[0xff, 0xd9]), "expected JPEG EOI marker");
    Ok(())
}

/**
 * Construct a multi-channel pointwise chain with clamps and repeated channel use.
 */
struct ChannelMixPipeline;

impl Operation for ChannelMixPipeline {
    /**
     * Add channel transforms whose order and intermediate clamp affect the result.
     */
    fn apply(&self, image: flint::sdk::ImageExpression) -> Result<flint::sdk::ImageExpression> {
        image
            .map_pixel(|pixel| {
                Pixel::rgb(
                    (pixel.r.clone() - 0.25).clamp(0.0, 1.0),
                    (pixel.g * 1.75 - 0.4).clamp(0.0, 1.0),
                    (pixel.b + pixel.r * 0.5).clamp(0.0, 1.0),
                )
            })?
            .map_pixel(|pixel| {
                Pixel::gray(pixel.r.clone() * pixel.r.clone() + pixel.r.clone() * pixel.g + pixel.r)
            })?
            .map_color(|value| (value - 0.1).max(0.0).min(1.0))
    }
}

/**
 * Build a bounded chain of pointwise stages for fusion and streaming checks.
 */
struct LongPointwisePipeline(usize);

impl Operation for LongPointwisePipeline {
    /**
     * Append deterministic affine stages while keeping the original graph public.
     */
    fn apply(&self, mut image: flint::sdk::ImageExpression) -> Result<flint::sdk::ImageExpression> {
        for index in 0..self.0 {
            let amount = (index % 5) as f32 * 0.007 - 0.01;
            image = image.map_color(|value| (value + amount).max(0.0).min(1.0))?;
        }
        Ok(image)
    }
}

/**
 * Build a zero-offset stencil to verify that its actual sampling is pointwise.
 */
struct ZeroOffsetStencil;

impl Operation for ZeroOffsetStencil {
    /**
     * Return each channel's value from the current pixel.
     */
    fn apply(&self, image: flint::sdk::ImageExpression) -> Result<flint::sdk::ImageExpression> {
        image.stencil(|neighborhood| neighborhood.at(0, 0))
    }
}

/**
 * Build a pointwise, zero-offset-stencil, pointwise chain.
 */
struct ZeroOffsetPipeline;

impl Operation for ZeroOffsetPipeline {
    /**
     * Place a zero-offset stencil between two arithmetic stages.
     */
    fn apply(&self, image: flint::sdk::ImageExpression) -> Result<flint::sdk::ImageExpression> {
        let image = image.map_color(|value| value + 0.05)?;
        let image = ZeroOffsetStencil.apply(image)?;
        image.map_color(|value| (value - 0.025).max(0.0).min(1.0))
    }
}

/**
 * Produce exceptional floating-point values through the public pixel expression API.
 */
struct ExceptionalValues;

impl Operation for ExceptionalValues {
    /**
     * Emit exceptional values, then apply a second stage before final packing.
     */
    fn apply(&self, image: flint::sdk::ImageExpression) -> Result<flint::sdk::ImageExpression> {
        image
            .map_pixel(|_| Pixel::rgb(f32::NAN, f32::INFINITY, -0.0))?
            .map_color(|value| (value - 0.25).max(-0.5))
    }
}

/**
 * Exercise fused kernels on both sides of a separable blur.
 */
struct MixedGaussianPipeline;

impl Operation for MixedGaussianPipeline {
    /**
     * Build pointwise groups before and after the horizontal and vertical passes.
     */
    fn apply(&self, image: ImageExpression) -> Result<ImageExpression> {
        let image = Brightness::new(0.17).apply(image)?;
        let image = image.map_color(|value| value * 1.125 - 0.2)?;
        let image = GaussianBlur::new(1.0).apply(image)?;
        let image = Grayscale.apply(image)?;
        Brightness::new(0.08).apply(image)
    }
}

/**
 * Keep constant arithmetic beside input samples in pointwise RGB programs.
 */
struct ConstantPointwise;

impl Operation for ConstantPointwise {
    /**
     * Retain one sample-dependent arithmetic instruction per output channel.
     */
    fn apply(&self, image: ImageExpression) -> Result<ImageExpression> {
        image.map_pixel(|pixel| {
            Pixel::rgb(
                pixel.r * (Expr::from(0.2) + Expr::from(0.3)),
                pixel.g + (Expr::from(0.8) - Expr::from(0.25)),
                pixel.b / (Expr::from(0.5) + Expr::from(0.5)),
            )
        })
    }
}

/**
 * Change channel count around pointwise stages with constant arithmetic.
 */
struct ConstantChannelPipeline;

impl Operation for ConstantChannelPipeline {
    /**
     * Propagate constant arithmetic across RGB, gray, and RGB stages.
     */
    fn apply(&self, image: ImageExpression) -> Result<ImageExpression> {
        image
            .map_pixel(|pixel| {
                Pixel::gray((pixel.r + pixel.g + pixel.b) / (Expr::from(1.0) + Expr::from(2.0)))
            })?
            .map_color(|value| value * (Expr::from(0.25) + Expr::from(0.75)))?
            .map_pixel(|pixel| {
                Pixel::rgb(
                    pixel.r.clone() + (Expr::from(0.1) + Expr::from(0.2)),
                    pixel.r.clone() * (Expr::from(0.5) + Expr::from(0.5)),
                    pixel.r - (Expr::from(0.8) - Expr::from(0.3)),
                )
            })
    }
}

/**
 * Produce a standalone spatial program with foldable constant arithmetic.
 */
struct ConstantStencil(i32, i32);

impl Operation for ConstantStencil {
    /**
     * Preserve the selected spatial sample while folding its coefficient.
     */
    fn apply(&self, image: ImageExpression) -> Result<ImageExpression> {
        let (dx, dy) = (self.0, self.1);
        image.stencil(|neighborhood| neighborhood.at(dx, dy) * (Expr::from(0.2) + Expr::from(0.3)))
    }
}

/**
 * Combine a folded spatial kernel with pointwise kernels on both sides.
 */
struct ConstantMixedPipeline;

impl Operation for ConstantMixedPipeline {
    /**
     * Preserve the spatial boundary and fuse compatible stages after it.
     */
    fn apply(&self, image: ImageExpression) -> Result<ImageExpression> {
        let image = ConstantPointwise.apply(image)?;
        let image = ConstantStencil(-1, 1).apply(image)?;
        ConstantChannelPipeline.apply(image)
    }
}

/**
 * Replace RGB input with three recursively constant output expressions.
 */
struct FullyConstantOutput;

impl Operation for FullyConstantOutput {
    /**
     * Produce constant RGB values without depending on the input samples.
     */
    fn apply(&self, image: ImageExpression) -> Result<ImageExpression> {
        image.map_pixel(|_| {
            Pixel::rgb(
                (Expr::from(0.2) + Expr::from(0.3)) * (Expr::from(2.0) - Expr::from(1.0)),
                (Expr::from(0.8) - Expr::from(0.3)) / (Expr::from(0.5) + Expr::from(0.5)),
                (Expr::from(0.1) + Expr::from(0.2))
                    .max(Expr::from(0.0))
                    .min(Expr::from(1.0)),
            )
        })
    }
}

/**
 * Return a write error on the first codec output attempt.
 */
struct FailingWriter;

impl Write for FailingWriter {
    /**
     * Fail the write so the encoder error reaches the public image API.
     */
    fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
        Err(std::io::Error::other("intentional test writer failure"))
    }

    /**
     * Accept flushes; writes are the deliberate failure point.
     */
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/**
 * Compare every backend and optimization setting for channel and exceptional pipelines.
 */
#[test]
fn optimized_pipeline_matches_disabled_bytes_for_all_backends() -> Result<()> {
    let pixels: Vec<u8> = (0..11 * 3)
        .map(|index| ((index * 73 + index * index * 11 + 29) % 256) as u8)
        .collect();
    let exceptional_pixel = [128, 64, 192];
    for backend in [Backend::Scalar, Backend::Simd] {
        assert_optimization_matches(&pixels, 11, 1, backend, || ChannelMixPipeline)?;
        assert_optimization_matches(&exceptional_pixel, 1, 1, backend, || ExceptionalValues)?;
        assert_optimization_matches(&pixels, 11, 1, backend, || ExceptionalValues)?;
        assert_optimization_matches(&pixels, 11, 1, backend, || LongPointwisePipeline(24))?;
        assert_optimization_matches(&pixels, 11, 1, backend, || ShiftLeftThenDown)?;
        assert_optimization_matches(&pixels, 11, 1, backend, || ZeroOffsetPipeline)?;
    }
    Ok(())
}

/**
 * Check that both Gaussian passes retain their spatial dependencies and that
 * reading through their ring windows yields the original bytes at every row.
 */
#[test]
fn mixed_gaussian_kernels_preserve_rows_and_boundaries() -> Result<()> {
    let (width, height) = (11, 18);
    let pixels: Vec<_> = (0..width * height * 3)
        .map(|position| (position * 73 + position / 13 * 19) as u8)
        .collect();
    for backend in [Backend::Scalar, Backend::Simd] {
        assert_optimization_matches(&pixels, width, height, backend, || MixedGaussianPipeline)?;
        let image = configured_raw_engine(backend, Optimization::Auto, 1_000_000)?
            .open_source(generated_source(width, height))?
            .apply(MixedGaussianPipeline)?;
        let plan = image.explain()?;
        let ranges: Vec<_> = plan
            .kernels
            .iter()
            .map(|kernel| kernel.stages.clone())
            .collect();
        assert_eq!(ranges, [0..2, 2..3, 3..4, 4..6]);
        assert_eq!(plan.kernels[1].min_x, -3);
        assert_eq!(plan.kernels[2].max_y, 3);
        assert_eq!(plan.kernels[2].buffered_rows, 7);
        let statistics = image.write_with_statistics("fraw", Box::new(std::io::sink()))?;
        assert_eq!(statistics.peak_managed_bytes, plan.managed_bytes);
    }
    Ok(())
}

/**
 * Check folded pointwise arithmetic against disabled optimization across SIMD tails.
 */
#[test]
fn constant_pointwise_folding_matches_backends_and_tails() -> Result<()> {
    for width in [1, 4, 5, 6, 7, 11] {
        let pixels: Vec<_> = (0..width * 3)
            .map(|index| ((index * 47 + 19) % 256) as u8)
            .collect();
        for backend in [Backend::Scalar, Backend::Simd] {
            assert_optimization_matches(&pixels, width as u32, 1, backend, || ConstantPointwise)?;
            let optimized = configured_raw_engine(backend, Optimization::Auto, 1_000_000)?
                .open_source(generated_source(width as u32, 1))?
                .apply(ConstantPointwise)?
                .explain()?;
            let disabled = configured_raw_engine(backend, Optimization::Disabled, 1_000_000)?
                .open_source(generated_source(width as u32, 1))?
                .apply(ConstantPointwise)?
                .explain()?;
            assert_eq!(optimized.kernels.len(), 1);
            assert_eq!(optimized.kernels[0].folded_instructions, 3);
            assert_eq!(disabled.kernels[0].folded_instructions, 0);
        }
    }
    Ok(())
}

/**
 * Verify standalone stencil folding keeps each horizontal, vertical, and mixed dependency.
 */
#[test]
fn constant_folding_covers_stencils_on_all_axes() -> Result<()> {
    let (width, height) = (7, 5);
    let pixels: Vec<_> = (0..width * height * 3)
        .map(|index| ((index * 31 + 7) % 256) as u8)
        .collect();
    for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1)] {
        let operation = || ConstantStencil(dx, dy);
        for backend in [Backend::Scalar, Backend::Simd] {
            assert_optimization_matches(&pixels, width as u32, height as u32, backend, operation)?;
        }
        let plan = configured_raw_engine(Backend::Scalar, Optimization::Auto, 1_000_000)?
            .open_source(generated_source(width as u32, height as u32))?
            .apply(operation())?
            .explain()?;
        assert_eq!(plan.kernels.len(), 1);
        assert_eq!(plan.kernels[0].folded_instructions, 3);
        assert_eq!((plan.kernels[0].min_x, plan.kernels[0].max_x), (dx, dx));
        assert_eq!((plan.kernels[0].min_y, plan.kernels[0].max_y), (dy, dy));
    }
    Ok(())
}

/**
 * Verify folding and channel propagation through a fused RGB-to-gray-to-RGB graph.
 */
#[test]
fn constant_folding_propagates_across_fused_channel_changes() -> Result<()> {
    let (width, height) = (7, 3);
    let pixels: Vec<_> = (0..width * height * 3)
        .map(|index| ((index * 59 + 13) % 256) as u8)
        .collect();
    for backend in [Backend::Scalar, Backend::Simd] {
        assert_optimization_matches(&pixels, width as u32, height as u32, backend, || {
            ConstantChannelPipeline
        })?;
        let image = configured_raw_engine(backend, Optimization::Auto, 1_000_000)?
            .open_source(generated_source(width as u32, height as u32))?
            .apply(ConstantChannelPipeline)?;
        let plan = image.explain()?;
        assert_eq!(plan.kernels.len(), 1);
        assert!(plan.kernels[0].folded_instructions >= 5);
        assert!(plan.kernels[0].instruction_count > 0);
        let statistics = image.write_with_statistics("fraw", Box::new(std::io::sink()))?;
        assert_eq!(statistics.kernel_count, 1);
        assert_eq!(
            statistics.folded_instructions,
            plan.kernels[0].folded_instructions
        );
        assert_eq!(statistics.peak_managed_bytes, plan.managed_bytes);
        assert!(plan.evaluator_workspace_bytes() <= plan.managed_bytes);
    }
    Ok(())
}

/**
 * Keep a folded stencil separate between folded pointwise kernels.
 */
#[test]
fn mixed_constant_folding_preserves_spatial_boundaries() -> Result<()> {
    let pixels: Vec<_> = (0..7 * 3 * 3).map(|index| (index * 23) as u8).collect();
    for backend in [Backend::Scalar, Backend::Simd] {
        assert_optimization_matches(&pixels, 7, 3, backend, || ConstantMixedPipeline)?;
        let plan = configured_raw_engine(backend, Optimization::Auto, 1_000_000)?
            .open_source(generated_source(7, 3))?
            .apply(ConstantMixedPipeline)?
            .explain()?;
        let ranges: Vec<_> = plan
            .kernels
            .iter()
            .map(|kernel| kernel.stages.clone())
            .collect();
        assert_eq!(ranges, [0..1, 1..2, 2..5]);
        assert_eq!(plan.kernels[1].min_x, -1);
        assert_eq!(plan.kernels[1].max_y, 1);
        assert!(
            plan.kernels
                .iter()
                .all(|kernel| kernel.folded_instructions > 0)
        );
    }
    Ok(())
}

/**
 * Ensure fully constant output still decodes, streams, accounts memory, and finalizes codecs.
 */
#[test]
fn fully_constant_output_preserves_streaming_and_codec_contracts() -> Result<()> {
    let width = 7;
    let engine = configured_raw_engine(Backend::Scalar, Optimization::Auto, 1_000_000)?;
    let short = engine
        .open_source(generated_source(width, 3))?
        .apply(FullyConstantOutput)?;
    let tall = engine
        .open_source(generated_source(width, 300))?
        .apply(FullyConstantOutput)?;
    let short_plan = short.explain()?;
    let tall_plan = tall.explain()?;
    assert_eq!(short_plan.kernels[0].folded_instructions, 9);
    assert_eq!(short_plan.kernels[0].instruction_count, 3);
    assert_eq!(short_plan.evaluator_workspace_bytes(), size_of::<f32>());
    assert_eq!(short_plan.managed_bytes, tall_plan.managed_bytes);

    let budget = short_plan.managed_bytes;
    let optimized_under_budget =
        configured_raw_engine(Backend::Scalar, Optimization::Auto, budget)?
            .open_source(generated_source(width, 3))?
            .apply(FullyConstantOutput)?;
    let original_under_budget =
        configured_raw_engine(Backend::Scalar, Optimization::Disabled, budget)?
            .open_source(generated_source(width, 3))?
            .apply(FullyConstantOutput)?;
    let original_plan = original_under_budget.explain()?;
    assert_eq!(original_plan.kernels[0].instruction_count, 21);
    assert!(original_plan.managed_bytes > budget);
    optimized_under_budget.write("fraw", Box::new(std::io::sink()))?;
    assert!(matches!(
        original_under_budget.write("fraw", Box::new(std::io::sink())),
        Err(Error::Limit(_))
    ));

    let capture = Capture::default();
    let statistics = tall.write_with_statistics(
        "fraw",
        Box::new(SharedWriter(capture.0.clone())) as Destination,
    )?;
    assert_eq!(capture.bytes().len(), 12 + width as usize * 300 * 3);
    assert!(
        capture.bytes()[12..]
            .as_chunks::<3>()
            .0
            .iter()
            .all(|pixel| *pixel == [128, 128, 77])
    );
    assert_eq!(statistics.peak_managed_bytes, tall_plan.managed_bytes);
    assert_eq!(
        statistics.folded_instructions,
        tall_plan.kernels[0].folded_instructions
    );
    assert!(statistics.peak_managed_bytes <= 1_000_000);

    let mut truncated = raw_image(width, 2, &[90; 2 * 7 * 3]);
    truncated.truncate(12 + width as usize * 3);
    let invalid = configured_raw_engine(Backend::Scalar, Optimization::Auto, 1_000_000)?
        .open_source(source(truncated, Arc::new(AtomicUsize::new(0))))?
        .apply(FullyConstantOutput)?;
    assert!(matches!(
        invalid.write("fraw", Box::new(std::io::sink())),
        Err(Error::Codec(_))
    ));
    Ok(())
}

/**
 * Check kernel ranges, dependencies, temporary workspace, and execution statistics.
 */
#[test]
fn execution_plan_reports_fusion_and_preserves_original_stages() -> Result<()> {
    let image = configured_raw_engine(Backend::Scalar, Optimization::Auto, 1_000_000)?
        .open_source(generated_source(7, 3))?
        .apply(Brightness::new(0.1))?
        .apply(Brightness::new(0.2))?;
    let plan = image.explain()?;
    assert_eq!(plan.stages.len(), 2);
    assert_eq!(plan.kernels.len(), 1);
    assert_eq!(plan.kernels[0].stages, 0..2);
    assert_eq!(plan.kernels[0].input, plan.source);
    assert_eq!(plan.kernels[0].output, plan.output);
    assert!(plan.kernels[0].instruction_count > 0);
    assert!(plan.kernels[0].temporary_values > 0);
    assert_eq!((plan.kernels[0].min_x, plan.kernels[0].max_x), (0, 0));
    assert_eq!((plan.kernels[0].min_y, plan.kernels[0].max_y), (0, 0));
    assert!(plan.kernels[0].buffered_rows > 0);
    assert!(plan.temporary_workspace_bytes() <= plan.managed_bytes);

    let stencil_image = configured_raw_engine(Backend::Scalar, Optimization::Auto, 1_000_000)?
        .open_source(generated_source(7, 3))?
        .apply(Brightness::new(0.1))?
        .apply(ShiftLeftThenDown)?
        .apply(Brightness::new(0.2))?;
    let stencil_plan = stencil_image.explain()?;
    assert_eq!(stencil_plan.stages.len(), 4);
    assert_eq!(stencil_plan.kernels.len(), 4);
    assert_eq!(stencil_plan.kernels[0].stages, 0..1);
    assert_eq!(stencil_plan.kernels[1].stages, 1..2);
    assert_eq!(stencil_plan.kernels[2].stages, 2..3);
    assert_eq!(stencil_plan.kernels[3].stages, 3..4);
    assert_eq!(stencil_plan.kernels[1].min_x, -1);
    assert_eq!(stencil_plan.kernels[2].max_y, 1);

    let zero_offset_plan = configured_raw_engine(Backend::Scalar, Optimization::Auto, 1_000_000)?
        .open_source(generated_source(7, 3))?
        .apply(Brightness::new(0.1))?
        .apply(ZeroOffsetStencil)?
        .explain()?;
    assert_eq!(zero_offset_plan.kernels.len(), 1);
    assert_eq!(zero_offset_plan.kernels[0].stages, 0..2);
    assert_eq!(zero_offset_plan.kernels[0].min_x, 0);
    assert_eq!(zero_offset_plan.kernels[0].max_y, 0);

    let capture = Capture::default();
    let statistics = image.write_with_statistics(
        "fraw",
        Box::new(SharedWriter(capture.0.clone())) as Destination,
    )?;
    assert_eq!(statistics.original_stage_count, 2);
    assert_eq!(statistics.kernel_count, 1);
    assert!(statistics.peak_managed_bytes >= plan.temporary_workspace_bytes());
    Ok(())
}

/**
 * Verify an over-budget plan stays inspectable and optimized working memory is bounded.
 */
#[test]
fn fused_memory_budget_is_height_independent_and_inspectable() -> Result<()> {
    let width = 9;
    let optimized_engine = configured_raw_engine(Backend::Scalar, Optimization::Auto, 1_000_000)?;
    let short = optimized_engine
        .open_source(generated_source(width, 32))?
        .apply(LongPointwisePipeline(24))?;
    let tall = optimized_engine
        .open_source(generated_source(width, 10_000))?
        .apply(LongPointwisePipeline(24))?;
    let short_plan = short.explain()?;
    let tall_plan = tall.explain()?;
    assert_eq!(short_plan.kernels.len(), 1);
    assert_eq!(short_plan.managed_bytes, tall_plan.managed_bytes);

    let disabled = configured_raw_engine(
        Backend::Scalar,
        Optimization::Disabled,
        short_plan.managed_bytes,
    )?
    .open_source(generated_source(width, 32))?
    .apply(LongPointwisePipeline(24))?;
    let disabled_plan = disabled.explain()?;
    assert!(disabled_plan.managed_bytes > short_plan.managed_bytes);
    assert!(matches!(
        disabled.write("fraw", Box::new(std::io::sink())),
        Err(Error::Limit(_))
    ));

    let budgeted = configured_raw_engine(
        Backend::Scalar,
        Optimization::Auto,
        short_plan.managed_bytes,
    )?
    .open_source(generated_source(width, 32))?
    .apply(LongPointwisePipeline(24))?;
    let budgeted_capture = Capture::default();
    let budgeted_statistics = budgeted.write_with_statistics(
        "fraw",
        Box::new(SharedWriter(budgeted_capture.0.clone())) as Destination,
    )?;
    assert!(budgeted_statistics.peak_managed_bytes <= short_plan.managed_bytes);

    let tall_capture = Capture::default();
    let tall_statistics = tall.write_with_statistics(
        "fraw",
        Box::new(SharedWriter(tall_capture.0.clone())) as Destination,
    )?;
    assert!(tall_statistics.peak_managed_bytes <= short_plan.managed_bytes);
    Ok(())
}

/**
 * Preserve codec failures with and without stages, including a fused graph.
 */
#[test]
fn direct_and_fused_paths_propagate_codec_errors() -> Result<()> {
    let mut truncated = raw_image(3, 2, &[80; 18]);
    truncated.truncate(12 + 3 * 3);
    for operation in [false, true] {
        let image = configured_raw_engine(Backend::Scalar, Optimization::Auto, 1_000_000)?
            .open_source(source(truncated.clone(), Arc::new(AtomicUsize::new(0))))?;
        let image = if operation {
            image
                .apply(Brightness::new(0.1))?
                .apply(Brightness::new(0.2))?
        } else {
            image
        };
        assert!(matches!(
            image.write("fraw", Box::new(std::io::sink())),
            Err(Error::Codec(_))
        ));

        let complete = configured_raw_engine(Backend::Scalar, Optimization::Auto, 1_000_000)?
            .open_source(generated_source(3, 2))?;
        let complete = if operation {
            complete
                .apply(Brightness::new(0.1))?
                .apply(Brightness::new(0.2))?
        } else {
            complete
        };
        assert!(matches!(
            complete.write("fraw", Box::new(FailingWriter)),
            Err(Error::Io(_)) | Err(Error::Codec(_))
        ));
    }
    Ok(())
}
