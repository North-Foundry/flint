# Writing a decoder

A decoder owns one sequential input session and exposes validated image metadata before pixel rows are requested. The runtime supplies each output buffer through `RowsMut`. Rows use packed Gray8 or interleaved RGB8 pixels, and each call fills a prefix of the supplied rows.

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

## Complete decoder

The example format contains the four byte magic `GRAY`, followed by little endian `u32` width and height, then exactly `width * height` Gray8 bytes.

```rust
use std::io::Read;

use flint_sdk::{Decoder, Error, ImageInfo, PixelFormat, Result, RowsMut, Source};

pub struct RawGrayDecoder {
    source: Source,
    info: ImageInfo,
    rows_left: usize,
}

impl Decoder for RawGrayDecoder {
    const NAME: &'static str = "raw-gray";
    const EXTENSIONS: &'static [&'static str] = &["gray"];

    fn probe(header: &[u8]) -> bool {
        header.starts_with(b"GRAY")
    }

    fn open(mut source: Source) -> Result<Self> {
        let mut header = [0_u8; 12];
        source.read_exact(&mut header)?;
        if &header[..4] != b"GRAY" {
            return Err(Error::Invalid("bad raw-gray signature".into()));
        }
        let width = u32::from_le_bytes(header[4..8].try_into().unwrap());
        let height = u32::from_le_bytes(header[8..12].try_into().unwrap());
        let info = ImageInfo::new(width, height, PixelFormat::Gray8)?;
        Ok(Self {
            source,
            rows_left: height as usize,
            info,
        })
    }

    fn info(&self) -> &ImageInfo {
        &self.info
    }

    fn read_rows(&mut self, mut output: RowsMut<'_>) -> Result<usize> {
        if output.info() != &self.info {
            return Err(Error::Invalid("decoder row metadata mismatch".into()));
        }
        let count = output.len().min(self.rows_left);
        for row in 0..count {
            self.source.read_exact(output.row_mut(row))?;
            self.rows_left -= 1;
        }
        Ok(count)
    }
}
```

`ImageInfo::new` rejects zero dimensions and unaddressable packed rows. `RowsMut::row_mut` excludes stride padding, so a decoder writes only packed pixel bytes. The runtime provides a nonempty buffer while rows remain. A zero return then signals end of image.

`read_exact` turns truncated pixel data into an I/O error. Implement `finish` when the format has a footer, checksum, or trailing validation that must be checked after the declared rows are read. Keep decoder session state inside the decoder and avoid retaining runtime row buffers.

## Register and use

Register the decoder with an engine to make the format available to an application.

```rust
use flint::Engine;
use flint_raw_gray::RawGrayDecoder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::builder()
        .register_decoder::<RawGrayDecoder>()
        .build()?;
    engine.open("input.gray")?.save("output.png")?;
    Ok(())
}
```
