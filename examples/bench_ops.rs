use std::{
    env,
    error::Error,
    io::{Cursor, Read, Write},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use flint::{
    Backend, Engine, Optimization,
    sdk::{
        Decoder, Destination, Encoder, Error as FlintError, ImageInformation, PixelFormat, Result,
        Rows, RowsMut, Source,
    },
};
use flint_basic::{Brightness, GaussianBlur, Grayscale};

const MAGIC: &[u8; 4] = b"FBR1";

/**
 * Run one deterministic in-process workload and report planning and execution metrics.
 *
 * The optional final argument selects `Auto` or `Disabled` optimization; omitting it
 * preserves the original command line and selects `Auto`.
 */
fn main() -> std::result::Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let width: u32 = args
        .next()
        .ok_or("usage: bench_ops WIDTH HEIGHT Scalar|Simd WORKLOAD ITERATIONS [Auto|Disabled]")?
        .parse()?;
    let height: u32 = args.next().ok_or("missing height")?.parse()?;
    let requested_backend = match args.next().as_deref() {
        Some("Scalar") => Backend::Scalar,
        Some("Simd") => Backend::Simd,
        Some(other) => return Err(format!("unknown backend: {other}").into()),
        None => return Err("missing backend".into()),
    };
    let operation = args.next().ok_or("missing operation")?;
    let iterations: usize = args.next().ok_or("missing iteration count")?.parse()?;
    let optimization = match args.next().as_deref() {
        None | Some("Auto") => Optimization::Auto,
        Some("Disabled") => Optimization::Disabled,
        Some(other) => return Err(format!("unknown optimization: {other}").into()),
    };
    if args.next().is_some() || iterations == 0 || width == 0 || height == 0 {
        return Err(
            "expected positive dimensions/iterations and at most one optimization mode".into(),
        );
    }

    let mut header = Vec::with_capacity(12);
    header.extend_from_slice(MAGIC);
    header.extend_from_slice(&width.to_le_bytes());
    header.extend_from_slice(&height.to_le_bytes());
    let engine = Engine::builder()
        .backend(requested_backend)
        .optimization(optimization)
        .register_decoder::<BenchDecoder>()
        .register_encoder::<BenchEncoder>()
        .build()?;
    let source_header = Arc::new(header);
    let image = engine.open_source({
        let source_header = Arc::clone(&source_header);
        move || Ok(Box::new(Cursor::new(source_header.as_ref().clone())) as Source)
    })?;
    let image = apply_workload(image, &operation)?;
    let planning_start = Instant::now();
    let plan = image.explain()?;
    let planning_seconds = planning_start.elapsed().as_secs_f64();
    let checksum = Arc::new(AtomicU64::new(0));

    image.write_with_statistics("bench", Box::new(ChecksumSink::new(Arc::clone(&checksum))))?;
    let start = Instant::now();
    let mut peak_managed_bytes = 0;
    let mut selected_backend = plan.backend;
    for _ in 0..iterations {
        let stats = image
            .write_with_statistics("bench", Box::new(ChecksumSink::new(Arc::clone(&checksum))))?;
        peak_managed_bytes = peak_managed_bytes.max(stats.peak_managed_bytes);
        selected_backend = stats.backend;
    }
    let elapsed = start.elapsed().as_secs_f64();
    let megapixels = f64::from(width) * f64::from(height) * iterations as f64 / 1_000_000.0;
    println!("elapsed_seconds={elapsed:.6}");
    println!("megapixels_per_second={:.3}", megapixels / elapsed);
    println!("iterations={iterations}");
    println!("width={width}");
    println!("height={height}");
    println!("operation={operation}");
    println!("optimization={optimization:?}");
    println!("requested_backend={requested_backend:?}");
    println!("backend={selected_backend:?}");
    println!("planning_seconds={planning_seconds:.9}");
    println!("stages={}", plan.stages.len());
    println!("kernels={}", plan.kernels.len());
    println!(
        "instruction_count={}",
        plan.kernels
            .iter()
            .map(|kernel| kernel.instruction_count)
            .sum::<usize>()
    );
    println!(
        "temporary_workspace_bytes={}",
        plan.temporary_workspace_bytes()
    );
    println!("peak_managed_bytes={peak_managed_bytes}");
    println!("estimated_managed_bytes={}", plan.managed_bytes);
    println!("checksum={:016x}", checksum.load(Ordering::Relaxed));
    Ok(())
}

/**
 * Add a named benchmark workload to an image expression.
 *
 * Chain workloads preserve explicit per-operation clamps by reusing `Brightness`.
 */
fn apply_workload(mut image: flint::Image, workload: &str) -> Result<flint::Image> {
    match workload {
        "grayscale" => image.apply(Grayscale),
        "brightness" => image.apply(Brightness::new(0.025)),
        "blur" => image.apply(GaussianBlur::new(2.0)),
        "combined" => image.apply(Grayscale)?.apply(GaussianBlur::new(2.0)),
        "brightness_grayscale_brightness" => image
            .apply(Brightness::new(0.025))?
            .apply(Grayscale)?
            .apply(Brightness::new(0.01)),
        "color_grayscale_contrast" => image
            .apply(ColorMix)?
            .apply(Grayscale)?
            .apply(Brightness::new(0.015))?
            .apply(Contrast),
        "chain_blur_chain" => {
            image = image.apply(Brightness::new(0.01))?.apply(ColorMix)?;
            image = image.apply(GaussianBlur::new(2.0))?;
            image.apply(ColorMix)?.apply(Brightness::new(0.01))
        }
        "mixed" => {
            image = image.apply(Brightness::new(0.025))?;
            image = image.apply(ColorMix)?;
            image = image.apply(GaussianBlur::new(2.0))?;
            image.apply(Brightness::new(0.01))
        }
        "color_mix" => image.apply(ColorMix),
        "expensive" => image.apply(ExpensiveColorMix),
        _ => {
            let (expensive, count) = workload
                .strip_prefix("expensive_chain")
                .map(|count| (true, count))
                .or_else(|| workload.strip_prefix("chain").map(|count| (false, count)))
                .ok_or_else(|| FlintError::Invalid(format!("unknown workload: {workload}")))?;
            let count: usize = count
                .parse()
                .map_err(|_| FlintError::Invalid("invalid chain length".into()))?;
            if !matches!(count, 2 | 5 | 10 | 20 | 50) {
                return Err(FlintError::Invalid(format!(
                    "unsupported chain length: {count}"
                )));
            }
            for index in 0..count {
                image = if expensive {
                    image.apply(ExpensiveColorMix)?
                } else {
                    image.apply(Brightness::new(0.004 + index as f32 * 0.0001))?
                };
            }
            Ok(image)
        }
    }
}

/**
 * Mixes RGB channels in a pointwise stage so fusion exercises cross-channel inputs.
 */
struct ColorMix;

impl flint::sdk::Operation for ColorMix {
    /**
     * Return RGB channels that each depend on all input channels.
     */
    fn apply(&self, image: flint::sdk::ImageExpression) -> Result<flint::sdk::ImageExpression> {
        image.map_pixel(|pixel| {
            flint::sdk::Pixel::rgb(
                pixel.r.clone() * 0.7 + pixel.g.clone() * 0.2 + pixel.b.clone() * 0.1,
                pixel.r.clone() * 0.1 + pixel.g.clone() * 0.7 + pixel.b.clone() * 0.2,
                pixel.r * 0.2 + pixel.g * 0.1 + pixel.b * 0.7,
            )
        })
    }
}

/**
 * Performs a more arithmetic intensive pointwise channel mix for fusion benchmarks.
 */
struct ExpensiveColorMix;

/**
 * Applies a contrast-like clamp independently to each color channel.
 */
struct Contrast;

impl flint::sdk::Operation for Contrast {
    /**
     * Return a pointwise contrast transformation with explicit clamping.
     */
    fn apply(&self, image: flint::sdk::ImageExpression) -> Result<flint::sdk::ImageExpression> {
        image.map_color(|sample| ((sample - 0.5) * 1.5 + 0.5).clamp(0.0, 1.0))
    }
}

impl flint::sdk::Operation for ExpensiveColorMix {
    /**
     * Build repeated weighted channel expressions while preserving channel sharing
     * opportunities for the optimizer.
     */
    fn apply(&self, image: flint::sdk::ImageExpression) -> Result<flint::sdk::ImageExpression> {
        image.map_pixel(|pixel| {
            let red = pixel.r.clone() * 0.7 + pixel.g.clone() * 0.2 + pixel.b.clone() * 0.1;
            let green = pixel.r.clone() * 0.1 + pixel.g.clone() * 0.7 + pixel.b.clone() * 0.2;
            let blue = pixel.r * 0.2 + pixel.g * 0.1 + pixel.b * 0.7;
            flint::sdk::Pixel::rgb(
                ((red.clone() * red.clone() + green.clone()) * 0.8 + blue.clone() * 0.2)
                    .clamp(0.0, 1.0),
                ((green.clone() * green.clone() + blue.clone()) * 0.8 + red.clone() * 0.2)
                    .clamp(0.0, 1.0),
                ((blue.clone() * blue + red) * 0.8 + green * 0.2).clamp(0.0, 1.0),
            )
        })
    }
}

/**
 * Generates deterministic RGB input rows without storing a complete image.
 */
struct BenchDecoder {
    info: ImageInformation,
    next_row: u32,
}

impl Decoder for BenchDecoder {
    const NAME: &'static str = "synthetic-bench";
    const EXTENSIONS: &'static [&'static str] = &["bench"];

    fn probe(header: &[u8]) -> bool {
        header.starts_with(MAGIC)
    }

    fn open(mut source: Source) -> Result<Self> {
        let mut header = [0; 12];
        source.read_exact(&mut header)?;
        if &header[..4] != MAGIC {
            return Err(FlintError::Invalid(
                "invalid benchmark source header".into(),
            ));
        }
        let width = u32::from_le_bytes(header[4..8].try_into().unwrap());
        let height = u32::from_le_bytes(header[8..12].try_into().unwrap());
        Ok(Self {
            info: ImageInformation::new(width, height, PixelFormat::Rgb8)?,
            next_row: 0,
        })
    }

    fn info(&self) -> &ImageInformation {
        &self.info
    }

    fn read_rows(&mut self, mut output: RowsMut<'_>) -> Result<usize> {
        let count = output
            .len()
            .min(self.info.height().saturating_sub(self.next_row) as usize);
        for row in 0..count {
            let y = self.next_row + row as u32;
            for (index, pixel) in output
                .row_mut(row)
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .enumerate()
            {
                let x = index as u32;
                pixel[0] = x.wrapping_mul(17).wrapping_add(y.wrapping_mul(3)) as u8;
                pixel[1] = x.wrapping_mul(5).wrapping_add(y.wrapping_mul(11)) as u8;
                pixel[2] = x.wrapping_mul(13).wrapping_add(y.wrapping_mul(7)) as u8;
            }
        }
        self.next_row += count as u32;
        Ok(count)
    }
}

/**
 * Consumes output rows through the benchmark checksum writer.
 */
struct BenchEncoder {
    destination: Destination,
    info: ImageInformation,
    rows_written: u32,
}

impl Encoder for BenchEncoder {
    const NAME: &'static str = "checksum-sink";
    const EXTENSIONS: &'static [&'static str] = &["bench"];
    type Options = ();

    fn open(destination: Destination, info: &ImageInformation, (): Self::Options) -> Result<Self> {
        Ok(Self {
            destination,
            info: *info,
            rows_written: 0,
        })
    }

    fn write_rows(&mut self, rows: Rows<'_>) -> Result<()> {
        if rows.info() != &self.info || rows.len() as u32 > self.info.height() - self.rows_written {
            return Err(FlintError::Invalid(
                "benchmark sink received invalid rows".into(),
            ));
        }
        for row in 0..rows.len() {
            self.destination.write_all(rows.row(row))?;
        }
        self.rows_written += rows.len() as u32;
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        if self.rows_written != self.info.height() {
            return Err(FlintError::Invalid(
                "benchmark sink received an incomplete image".into(),
            ));
        }
        self.destination.flush()?;
        Ok(())
    }
}

/**
 * Hashes encoded output bytes to keep the benchmark's result observable.
 */
struct ChecksumSink {
    checksum: Arc<AtomicU64>,
    state: u64,
}

impl ChecksumSink {
    /**
     * Create a sink that publishes its completed FNV-1a checksum to shared state.
     */
    fn new(checksum: Arc<AtomicU64>) -> Self {
        Self {
            checksum,
            state: 0xcbf29ce484222325,
        }
    }
}

impl Write for ChecksumSink {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        for &byte in bytes {
            self.state = (self.state ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.checksum.store(self.state, Ordering::Relaxed);
        Ok(())
    }
}
