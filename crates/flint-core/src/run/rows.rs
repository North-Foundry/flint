use super::buffer;
use crate::plan::float_ring_bytes;
use flint_sdk::{ImageInformation, Result};

/**
 * Normalize packed byte samples into a planar floating-point row.
 */
pub(super) fn unpack(bytes: &[u8], info: ImageInformation, row: &mut [f32]) {
    let width = info.width() as usize;
    for (channel, plane) in row.chunks_exact_mut(width).enumerate() {
        for (value, sample) in plane
            .iter_mut()
            .zip(bytes[channel..].iter().step_by(info.format().channels()))
        {
            *value = f32::from(*sample) / 255.0;
        }
    }
}

/**
 * Quantize a planar floating-point row into packed byte samples.
 */
pub(super) fn pack(row: &[f32], info: ImageInformation, bytes: &mut [u8]) {
    let width = info.width() as usize;
    for (channel, plane) in row.chunks_exact(width).enumerate() {
        for (sample, value) in bytes[channel..]
            .iter_mut()
            .step_by(info.format().channels())
            .zip(plane)
        {
            *sample = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
}

/**
 * A fixed-size, circular window of planar rows.
 *
 * An absolute row number is mapped to one of `capacity` physical slots with
 * `index % capacity`. The ring never stores the whole image: each stage keeps
 * only the vertical span its stencil can read, and the plan guarantees that a
 * slot is overwritten only after every consumer has finished with that row.
 */
pub(super) struct Ring {
    info: ImageInformation,
    capacity: usize,
    data: Vec<f32>,
}

impl Ring {
    /**
     * Allocate a zeroed ring for the image layout and row capacity.
     *
     * `capacity` is the number of rows in the window and must be nonzero. Each
     * row contains `width * channels` `f32` samples. Size arithmetic is checked
     * by `float_ring_bytes`; allocation failures become `Error::Limit`.
     */
    pub(super) fn new(info: ImageInformation, capacity: usize) -> Result<Self> {
        Ok(Self {
            info,
            capacity,
            data: buffer(float_ring_bytes(info, capacity)? / size_of::<f32>(), 0.0)?,
        })
    }

    /**
     * Return the image metadata that determines the layout of each row.
     */
    pub(super) fn info(&self) -> ImageInformation {
        self.info
    }

    /**
     * Return the allocated capacity in bytes for memory accounting.
     */
    pub(super) fn capacity_bytes(&self) -> usize {
        self.data.capacity() * size_of::<f32>()
    }

    /**
     * Map an absolute row number to its physical slot range.
     *
     * With capacity 3, logical rows 0, 1, 2 use slots 0, 1, 2 and logical row
     * 3 reuses slot 0:
     *
     * ```text
     * logical row:   0 1 2 3 4 5
     * physical slot: 0 1 2 0 1 2
     * ```
     *
     * The plan's vertical span guarantees row 0 is no longer needed when row 3
     * is written.
     */
    fn row_range(&self, index: usize) -> std::ops::Range<usize> {
        // A packed byte row contains one sample per byte; the planar f32 row
        // keeps the same sample count and layout.
        let row_length = self.info.row_bytes();
        let slot = index % self.capacity;
        let start = slot * row_length;
        start..start + row_length
    }

    /**
     * Borrow the logical row at `index` from its physical slot.
     */
    pub(super) fn row(&self, index: usize) -> &[f32] {
        &self.data[self.row_range(index)]
    }

    /**
     * Borrow the logical row at `index` for writing into its physical slot.
     */
    pub(super) fn row_mut(&mut self, index: usize) -> &mut [f32] {
        let range = self.row_range(index);
        &mut self.data[range]
    }
}
