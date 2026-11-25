# Flint

Flint is a Rust image processing library from NorthFoundry. It builds a lazy image graph and evaluates it one row at a time when you write the output.

```rust
use flint::Image;

fn main() -> flint::Result<()> {
    Image::open("input.png")?.save("output.jpg")
}
```

Operations compose through the same API.

```rust
use flint::{Engine, Result};
use flint_basic::{GaussianBlur, Grayscale};

fn main() -> Result<()> {
    let engine = Engine::builder().memory_limit(32 * 1024 * 1024).build()?;
    let image = engine
        .open("input.png")?
        .apply(Grayscale)?
        .apply(GaussianBlur::new(2.0))?;
    println!("{}", image.explain()?);
    image.save("output.jpg")
}
```

## Supported images

Input is detected by signature. The destination extension selects the encoder, ignoring ASCII case. The `png` and `jpeg` Cargo features are enabled by default and can be selected individually.

PNG input and output support non-interlaced Gray8 and RGB8. The PNG decoder rejects transparency, animation, indexed color, and other sample depths. Its limits include a 16 MiB packed row and a 1 MiB EXIF chunk. Conversion discards color profiles and other ancillary metadata.

JPEG output is baseline sequential at quality 85 by default, with 4:2:0 sampling for color. JPEG input is deferred. The bundled JPEG encoder needs a C compiler and enables MozJPEG's NEON support on AArch64 toolchains that support it.

The default managed memory budget is 64 MiB and covers runtime pixel buffers and evaluator scratch space. `explain()` estimates those bytes. `save_with_statistics()` and `write_with_statistics()` report measured peak buffer capacity and the selected backend after a successful run. Codec allocations and process RSS are separate measurements.

## Build and extend

```sh
cargo build --workspace --locked
cargo test --workspace --locked
cargo run -p flint --example convert -- input.png output.jpg
cargo run -p flint --example independent_operation -- input.png output.png
python3 benchmarks/run.py --output-dir benchmarks/output
```

[Architecture](.agents/docs/architecture.md) · [Write an operation](.agents/docs/writing-an-operation.md) · [Write a decoder](.agents/docs/writing-a-decoder.md) · [Write an encoder](.agents/docs/writing-an-encoder.md) · [Benchmarks](benchmarks/README.md)
