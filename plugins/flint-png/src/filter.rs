use flint_sdk::{Error, PixelFormat, Result};

pub(super) fn unfilter(
    filter: u8,
    raw: &[u8],
    previous: &[u8],
    format: PixelFormat,
    output: &mut [u8],
) -> Result<()> {
    match format {
        PixelFormat::Gray8 => reconstruct::<1>(filter, raw, previous, output),
        PixelFormat::Rgb8 => reconstruct::<3>(filter, raw, previous, output),
    }
}

fn reconstruct<const CHANNELS: usize>(
    filter: u8,
    raw: &[u8],
    previous: &[u8],
    output: &mut [u8],
) -> Result<()> {
    match filter {
        0 => output.copy_from_slice(raw),
        1 => predict::<CHANNELS>(raw, previous, output, |left, _, _| left),
        2 => {
            for ((out, &sample), &above) in output.iter_mut().zip(raw).zip(previous) {
                *out = sample.wrapping_add(above);
            }
        }
        3 => predict::<CHANNELS>(raw, previous, output, |left, above, _| {
            ((u16::from(left) + u16::from(above)) / 2) as u8
        }),
        4 => predict::<CHANNELS>(raw, previous, output, paeth),
        _ => return Err(Error::Codec(format!("unknown PNG row filter {filter}"))),
    }
    Ok(())
}

fn predict<const CHANNELS: usize>(
    raw: &[u8],
    previous: &[u8],
    output: &mut [u8],
    predictor: impl Fn(u8, u8, u8) -> u8,
) {
    let mut left = [0; CHANNELS];
    let mut upper_left = [0; CHANNELS];
    let (output, _) = output.as_chunks_mut::<CHANNELS>();
    let (raw, _) = raw.as_chunks::<CHANNELS>();
    let (previous, _) = previous.as_chunks::<CHANNELS>();
    for ((out, raw), above) in output.iter_mut().zip(raw).zip(previous) {
        for c in 0..CHANNELS {
            let sample = raw[c].wrapping_add(predictor(left[c], above[c], upper_left[c]));
            out[c] = sample;
            left[c] = sample;
            upper_left[c] = above[c];
        }
    }
}

#[inline]
fn paeth(left: u8, above: u8, upper_left: u8) -> u8 {
    let pa = i16::from(above) - i16::from(upper_left);
    let pb = i16::from(left) - i16::from(upper_left);
    let pc = (pa + pb).abs();
    let (pa, pb) = (pa.abs(), pb.abs());
    let best = if pb < pa { above } else { left };
    if pc < pa.min(pb) { upper_left } else { best }
}
