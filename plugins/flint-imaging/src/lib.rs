/*!
 * Flint's official image-processing operations.
 *
 * Operations are grouped by semantic category and re-exported from the crate
 * root for convenient imports.
 */

pub mod blur;
pub mod color;

pub use blur::GaussianBlur;
pub use color::{Brightness, Grayscale};
