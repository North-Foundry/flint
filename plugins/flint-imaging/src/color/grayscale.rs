/*!
 * Grayscale conversion.
 */

use flint_sdk::{ImageExpression, Operation, Pixel, Result};

/**
 * Converts RGB pixels to grayscale using encoded-space luma coefficients.
 * The weights apply directly to the stored RGB samples.
 */
#[derive(Clone, Copy, Debug, Default)]
pub struct Grayscale;

impl Operation for Grayscale {
    /**
     * Replaces each pixel with its weighted luma sample.
     */
    fn apply(&self, image: ImageExpression) -> Result<ImageExpression> {
        image.map_pixel(|p| Pixel::gray(p.r * 0.2126 + p.g * 0.7152 + p.b * 0.0722))
    }
}
