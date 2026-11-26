/*!
 * Gaussian blur.
 */

use flint_sdk::{Error, Expr, ImageExpression, Operation, Result};

/**
 * Applies a separable Gaussian blur, using a finite radius of `ceil(3σ)`.
 */
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianBlur {
    sigma: f32,
}

impl GaussianBlur {
    /**
     * Creates a Gaussian blur with the given standard deviation in pixels.
     */
    pub const fn new(sigma: f32) -> Self {
        Self { sigma }
    }

    /**
     * Resolves the stencil radius for the configured sigma.
     *
     * Returns an error when sigma is not finite or not greater than zero, and
     * when the radius exceeds the SDK's maximum stencil offset.
     */
    fn radius(&self) -> Result<i32> {
        if !self.sigma.is_finite() || self.sigma <= 0.0 {
            return Err(Error::Invalid(
                "Gaussian blur sigma must be finite and greater than zero".into(),
            ));
        }

        // f64 multiplication avoids overflowing for finite f32 sigma values.
        // At the stencil limit, 4 * (2r + 1) - 1 is 32,771 instructions,
        // below the SDK's 65,536-instruction limit for each pass.
        let radius = (3.0 * f64::from(self.sigma)).ceil();
        if radius > f64::from(flint_sdk::MAX_STENCIL_OFFSET) {
            return Err(Error::Limit(
                "Gaussian blur radius exceeds the maximum stencil offset".into(),
            ));
        }

        Ok(radius as i32)
    }

    /**
     * Builds normalized one-dimensional Gaussian weights for `radius`.
     */
    fn weights(&self, radius: i32) -> Vec<f32> {
        let sigma = self.sigma as f64;
        let denom = 2.0 * sigma * sigma;
        let weights: Vec<f64> = (-radius..=radius)
            .map(|offset| (-(offset as f64).powi(2) / denom).exp())
            .collect();
        let sum: f64 = weights.iter().sum();
        weights
            .into_iter()
            .map(|weight| (weight / sum) as f32)
            .collect()
    }
}

impl Operation for GaussianBlur {
    /**
     * Composes a horizontal and a vertical stencil stage from one kernel.
     */
    fn apply(&self, image: ImageExpression) -> Result<ImageExpression> {
        let radius = self.radius()?;
        let weights = self.weights(radius);

        let image = image.stencil(|n| {
            (-radius..=radius)
                .map(|x| n.at(x, 0) * weights[(x + radius) as usize])
                .sum::<Expr>()
        })?;

        image.stencil(|n| {
            (-radius..=radius)
                .map(|y| n.at(0, y) * weights[(y + radius) as usize])
                .sum::<Expr>()
        })
    }
}
