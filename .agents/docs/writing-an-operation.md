# Writing an operation

An operation implements `flint_sdk::Operation` and builds an immutable expression graph. The closures passed to `ImageExpression` primitives run once during graph construction. Flint evaluates their expressions over image samples when a consumer saves or writes the image.

## Crate setup

For an extension crate placed at `plugins/flint-tint`, add this dependency.

```toml
[package]
name = "flint-tint"
version = "0.1.0"
edition = "2024"

[dependencies]
flint-sdk = { path = "../../crates/flint-sdk" }
```

## Complete operation

This operation adds a different normalized amount to each RGB channel. `map_pixel` also declares the output pixel format through `Pixel::rgb`.

```rust
use flint_sdk::{ImageExpression, Operation, Pixel, Result};

pub struct Tint {
    pub red: f32,
    pub green: f32,
    pub blue: f32,
}

impl Operation for Tint {
    fn apply(&self, image: ImageExpression) -> Result<ImageExpression> {
        image.map_pixel(|p| {
            Pixel::rgb(
                (p.r + self.red).clamp(0.0, 1.0),
                (p.g + self.green).clamp(0.0, 1.0),
                (p.b + self.blue).clamp(0.0, 1.0),
            )
        })
    }
}
```

Use `map_color` when one expression should run independently for each channel. Use `map_pixel` when output channels depend on named input channels or the operation changes the pixel format. Use `stencil` for relative sample access through `n.at(dx, dy)`. Its initial border rule is clamp to edge.

`Expr` supports `f32` arithmetic, `min`, `max`, `clamp`, and iterator summation. Keep intermediate calculations in the expression graph. The runtime performs final pixel quantization after the image stages.

## Consumer use

Add the facade and the operation crate to an application, then apply the operation through the normal image API.

```rust
use flint::Image;
use flint_tint::Tint;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    Image::open("input.png")?
        .apply(Tint {
            red: 0.05,
            green: 0.0,
            blue: 0.0,
        })?
        .save("output.png")?;
    Ok(())
}
```

`ImageExpression::metadata()` returns the current `ImageInformation` by reference. Use it to inspect dimensions or format when choosing an algorithm. Return SDK errors for invalid parameters, and keep pixel iteration, buffers, and execution policy in the runtime.
