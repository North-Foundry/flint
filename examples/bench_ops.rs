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
    Backend, Engine,
    sdk::{
        Decoder, Destination, Encoder, Error as FlintError, ImageInformation, PixelFormat, Result,
        Rows, RowsMut, Source,
    },
};
use flint_basic::{GaussianBlur, Grayscale};

const MAGIC: &[u8; 4] = b"FBR1";

fn main() -> std::result::Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let width: u32 = args
        .next()
        .ok_or("usage: bench_ops WIDTH HEIGHT Scalar|Simd grayscale|blur|combined ITERATIONS")?
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
    if args.next().is_some() || iterations == 0 {
        return Err("expected positive iterations and no extra arguments".into());
    }

    let mut header = Vec::with_capacity(12);
    header.extend_from_slice(MAGIC);
    header.extend_from_slice(&width.to_le_bytes());
    header.extend_from_slice(&height.to_le_bytes());
    let engine = Engine::builder()
        .backend(requested_backend)
        .register_decoder::<BenchDecoder>()
        .register_encoder::<BenchEncoder>()
        .build()?;
    let source_header = Arc::new(header);
    let image = engine.open_source({
        let source_header = Arc::clone(&source_header);
        move || Ok(Box::new(Cursor::new(source_header.as_ref().clone())) as Source)
    })?;
    let image = match operation.as_str() {
        "grayscale" => image.apply(Grayscale)?,
        "blur" => image.apply(GaussianBlur::new(2.0))?,
        "combined" => image.apply(Grayscale)?.apply(GaussianBlur::new(2.0))?,
        _ => return Err(format!("unknown operation: {operation}").into()),
    };
    let plan = image.explain()?;
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
    println!("requested_backend={requested_backend:?}");
    println!("backend={selected_backend:?}");
    println!("peak_managed_bytes={peak_managed_bytes}");
    println!("estimated_managed_bytes={}", plan.managed_bytes);
    println!("checksum={:016x}", checksum.load(Ordering::Relaxed));
    Ok(())
}

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

struct ChecksumSink {
    checksum: Arc<AtomicU64>,
    state: u64,
}

impl ChecksumSink {
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
