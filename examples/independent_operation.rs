//! A third-party style operation using only the public SDK contracts.

use flint::Image;
use flint_sdk::{ImageExpr, Operation, Result};

/// Inverts normalized RGB color channels.
struct Invert;

impl Operation for Invert {
    fn apply(&self, image: ImageExpr) -> Result<ImageExpr> {
        image.map_color(|channel| 1.0 - channel)
    }
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    Image::open("input.png")?
        .apply(Invert)?
        .save("output.png")?;
    Ok(())
}
