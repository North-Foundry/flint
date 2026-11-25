//! Basic image operations built only from Flint's public SDK.

use flint_sdk::{Error, Expr, ImageExpression, Operation, Pixel, Result};

/// Adds an amount to every normalized color channel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Brightness {
    amount: f32,
}

impl Brightness {
    pub const fn new(amount: f32) -> Self {
        Self { amount }
    }
}

impl Operation for Brightness {
    fn apply(&self, image: ImageExpression) -> Result<ImageExpression> {
        image.map_color(|x| (x + self.amount).clamp(0.0, 1.0))
    }
}

/// Converts RGB pixels to grayscale using encoded-space luma coefficients.
/// The weights apply directly to the stored RGB samples.
#[derive(Clone, Copy, Debug, Default)]
pub struct Grayscale;

impl Operation for Grayscale {
    fn apply(&self, image: ImageExpression) -> Result<ImageExpression> {
        image.map_pixel(|p| Pixel::gray(p.r * 0.2126 + p.g * 0.7152 + p.b * 0.0722))
    }
}

/// Applies a separable Gaussian blur, using a finite radius of `ceil(3σ)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianBlur {
    sigma: f32,
}

impl GaussianBlur {
    pub const fn new(sigma: f32) -> Self {
        Self { sigma }
    }

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
