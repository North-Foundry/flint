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
 * Store a bounded window of planar rows using circular indexing.
 */
pub(super) struct Ring {
    info: ImageInformation,
    capacity: usize,
    data: Vec<f32>,
}

impl Ring {
    /**
     * Allocate the ring for the specified row format and capacity.
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

    pub(super) fn capacity_bytes(&self) -> usize {
        self.data.capacity() * size_of::<f32>()
    }

    fn row_range(&self, index: usize) -> std::ops::Range<usize> {
        let len = self.info.row_bytes(); // A packed byte row contains one byte per sample.
        let start = (index % self.capacity) * len;
        start..start + len
    }

    pub(super) fn row(&self, index: usize) -> &[f32] {
        &self.data[self.row_range(index)]
    }

    pub(super) fn row_mut(&mut self, index: usize) -> &mut [f32] {
        let range = self.row_range(index);
        &mut self.data[range]
    }
}
