/*!
 * Per-channel brightness adjustment.
 */

use flint_sdk::{ImageExpression, Operation, Result};

/**
 * Adds an amount to every normalized color channel.
 */
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Brightness {
    amount: f32,
}

impl Brightness {
    /**
     * Creates a brightness adjustment that adds `amount` to each channel.
     *
     * The adjusted channel is clamped to the normalized color interval during
     * evaluation.
     */
    pub const fn new(amount: f32) -> Self {
        Self { amount }
    }
}

impl Operation for Brightness {
    /**
     * Maps every active channel through the brightness adjustment.
     */
    fn apply(&self, image: ImageExpression) -> Result<ImageExpression> {
        image.map_color(|x| (x + self.amount).clamp(0.0, 1.0))
    }
}
