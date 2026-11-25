# Writing an encoder

An encoder owns one sequential destination session. `open` receives the final image metadata and defaultable options. `write_rows` consumes borrowed packed rows in top to bottom order. `finish` validates completeness and reports errors from final output and flushing.

## Crate setup

For an extension crate placed at `plugins/flint-raw-gray`, add this dependency.

```toml
[package]
name = "flint-raw-gray"
version = "0.1.0"
edition = "2024"

[dependencies]
flint-sdk = { path = "../../crates/flint-sdk" }
```

## Complete encoder

This encoder writes the `GRAY` format described in the decoder guide. It accepts Gray8 data and writes a small header before the pixel rows.

```rust
use std::io::Write;

use flint_sdk::{Destination, Encoder, Error, ImageInfo, PixelFormat, Result, Rows};

pub struct RawGrayEncoder {
    destination: Destination,
    info: ImageInfo,
    rows_written: usize,
}

impl Encoder for RawGrayEncoder {
    const NAME: &'static str = "raw-gray";
    const EXTENSIONS: &'static [&'static str] = &["gray"];
    type Options = ();

    fn open(
        mut destination: Destination,
        info: &ImageInfo,
        (): Self::Options,
    ) -> Result<Self> {
        if info.format() != PixelFormat::Gray8 {
            return Err(Error::Unsupported(
                "raw-gray encoder accepts Gray8 images".into(),
            ));
        }
        destination.write_all(b"GRAY")?;
        destination.write_all(&info.width().to_le_bytes())?;
        destination.write_all(&info.height().to_le_bytes())?;
        Ok(Self {
            destination,
            info: *info,
            rows_written: 0,
        })
    }

    fn write_rows(&mut self, rows: Rows<'_>) -> Result<()> {
        let height = self.info.height() as usize;
        if rows.info() != &self.info
            || self.rows_written > height
            || rows.len() > height - self.rows_written
        {
            return Err(Error::Invalid(
                "encoder rows exceed or differ from image metadata".into(),
            ));
        }
        for row in 0..rows.len() {
            self.destination.write_all(rows.row(row))?;
        }
        self.rows_written += rows.len();
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        if self.rows_written != self.info.height() as usize {
            return Err(Error::Invalid("encoder received an incomplete image".into()));
        }
        self.destination.flush()?;
        Ok(())
    }
}
```

`Rows::row` excludes stride padding. Check each batch against the metadata and remaining row count before writing it. `finish` is required for successful output and is the place to write trailers, flush buffered codecs, and report finalization errors.

## Register and use

An application can register the encoder on an engine. The destination extension selects it when saving.

```rust
use flint::Engine;
use flint_raw_gray::RawGrayEncoder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::builder()
        .register_encoder::<RawGrayEncoder>()
        .build()?;
    engine.open("input.png")?.save("output.gray")?;
    Ok(())
}
```
