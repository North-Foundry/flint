# Benchmarks

Run the standard-library-only harness from the workspace root.

```sh
python3 benchmarks/run.py --output-dir benchmarks/output --iterations 3
```

It writes `results.json` and generates all fixtures deterministically. The end-to-end matrix converts RGB8 PNG files to JPEG at 640x480 and 1920x1080 with gradient, checker, and integer-hash-noise content. It also compares fixed-width PNGs at 256x512, 256x2048, and 256x8192 (`--width` changes the width for that series). Each converter run is a fresh process; time includes process startup, PNG decode, JPEG encode, and filesystem I/O. The JSON stores per-run wall time, megapixels/s, peak RSS, and JPEG byte size.

Flint conversion rows additionally include the `save_with_stats` measured peak managed bytes, the execution-plan estimate, and the backend returned by execution. Other converters leave Flint-specific fields null.

libvips (`vips`), ImageMagick (`magick` or `convert`), and PHP with GD are discovered independently. Missing executables or PHP GD are skipped. Flint's `convert` example is built with `crates/flint/Cargo.toml` and run directly. The examples `examples/convert.rs` and `examples/bench_ops.rs` are registered there.

## Codec settings

Flint's MozJPEG-backed encoder defaults to quality 85 and RGB 4:2:0 chroma sampling. The harness requests quality 85 and 4:2:0 from libvips and ImageMagick. PHP GD uses quality 85 and its default chroma sampling. JPEG bytes can differ by library despite the shared quality setting. The harness records settings and detected tool versions with the results.

For a direct custom Flint conversion command, pass an argument list using `{input}` and `{output}` placeholders with `--flint-convert`. For custom in-process adapters, `--flint-ops` accepts `{width}`, `{height}`, `{backend}`, `{operation}`, and `{iterations}`.

## In-process operations

`bench_ops` registers a benchmark-only raw decoder and sink encoder through the public SDK contracts. Its source contains only a 12-byte dimensions header; RGB8 pixel rows are generated deterministically as the runtime requests them. The sink consumes output rows into a checksum. No complete image is held, and no PNG, JPEG, file, or compression work occurs in this benchmark.

The example builds the graph and records its execution plan, then performs one untimed warmup before timing. The timed loop calls `Image::write_with_stats("bench", sink)` repeatedly, so it measures streaming decode, the selected operation pipeline, quantization, and checksum-sink consumption in one Flint process. The public write call analyzes the plan for each execution, and that small planning cost is included. It reports `elapsed_seconds`, throughput, checksum, requested backend, actual selected backend (including scalar fallback), measured `peak_managed_bytes` (the maximum returned by timed writes), and estimated `estimated_managed_bytes` (the plan estimate). The harness also reports process peak RSS separately; this includes the Rust process and its runtime/allocator.

The internal matrix uses deterministic synthetic rows at 640x480, 1920x1080, and the same fixed-width height series as the conversion benchmark. It runs Grayscale, Gaussian Blur (sigma 2), and Grayscale followed by Gaussian Blur for requested Scalar and SIMD backends. Graph construction and explicit plan inspection happen before timing. Each timed execution includes the runtime's plan preparation and allocation. Each invocation is a separate process to keep peak RSS per result.

## Interpreting results

- `peak_managed_bytes`, `estimated_managed_bytes`, and `selected_backend` are populated for Flint rows; external tools report `null` for them.
- Compare the three fixed-width heights to see memory scaling; use the same `--width`, build profile, tool versions, and host when comparing runs.
- End-to-end and in-process timings cover different work and should be compared separately.
- Peak RSS uses Python's `resource.getrusage(RUSAGE_CHILDREN)` in an isolated worker. Linux values are converted from KiB and macOS values are already bytes. Platforms without `resource` report `null`.
- PNG payload bytes can vary with Python/zlib versions, but decoded fixture pixels are deterministic.
- No results are implied by the harness; report only runs actually performed.
