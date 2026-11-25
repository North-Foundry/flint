use crate::{
    Backend, ExecutionPlan, StagePlan,
    plan::{check_budget, float_ring_bytes, graph_stages},
    registry::{DecoderSession, SelectedDecoder, SelectedEncoder},
};
use flint_sdk::{
    Destination, Error, Expr, ImageExpr, ImageInfo, Result, Rows, RowsMut, Source,
    graph::{Instruction, Stage},
};
use wide::f32x4;

/// Observations from a completed execution.
#[derive(Clone, Copy, Debug)]
pub struct ExecutionStats {
    pub backend: Backend,
    /// Peak total capacity of managed pixel and evaluator buffers, in bytes.
    pub peak_managed_bytes: usize,
}

pub fn execute(
    image: &ImageExpr,
    backend: Backend,
    memory_limit: usize,
    source: &dyn Fn() -> Result<Source>,
    decoder: SelectedDecoder,
    encoder: SelectedEncoder,
    destination: Destination,
) -> Result<ExecutionStats> {
    let plan = ExecutionPlan::analyze(image, decoder.name(), backend)?;
    check_budget(plan.managed_bytes, memory_limit)?;
    let mut decoder = decoder.session(source()?)?;
    if *decoder.info() != plan.source {
        return Err(Error::Invalid(
            "source metadata changed since the image was opened".into(),
        ));
    }
    let mut input_bytes = buffer(plan.source.row_bytes(), 0u8)?;

    let height = plan.source.height() as usize;
    if plan.stages.is_empty() {
        let stats = stats(plan.backend, input_bytes.capacity(), memory_limit)?;
        let mut encoder = encoder.session(destination, &plan.output)?;
        for _ in 0..height {
            decode_row(&mut *decoder, &plan.source, &mut input_bytes)?;
            encoder.write_rows(Rows::new(&plan.output, input_bytes.len(), &input_bytes)?)?;
        }
        decoder.finish()?;
        encoder.finish()?;
        return Ok(stats);
    }

    let mut source = FloatSource {
        decoder,
        pixels: Ring::new(plan.source, plan.stages[0].buffered_rows)?,
        bytes: input_bytes,
        next: 0,
    };
    let mut output_bytes = buffer(plan.output.row_bytes(), 0u8)?;
    let mut stages = graph_stages(image)
        .zip(plan.stages.iter().enumerate().rev())
        .map(|((stage, _), (i, analysis))| {
            let rows = plan.stages.get(i + 1).map_or(1, |next| next.buffered_rows);
            Ok(StageState {
                stage,
                analysis,
                pixels: Ring::new(analysis.output, rows)?,
                next: 0,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    stages.reverse();
    let stack_depth = plan
        .stages
        .iter()
        .map(|stage| stage.stack_depth)
        .max()
        .unwrap_or(0);
    let mut scratch = Scratch {
        scalar: buffer(stack_depth, 0.0)?,
        vector: if plan.backend == Backend::Simd {
            buffer(stack_depth, f32x4::ZERO)?
        } else {
            Vec::new()
        },
    };
    // These buffers coexist throughout processing. Their capacities never change.
    let peak_managed_bytes = source.bytes.capacity()
        + output_bytes.capacity()
        + source.pixels.capacity_bytes()
        + stages
            .iter()
            .map(|stage| stage.pixels.capacity_bytes())
            .sum::<usize>()
        + scratch.scalar.capacity() * size_of::<f32>()
        + scratch.vector.capacity() * size_of::<f32x4>();
    let stats = stats(plan.backend, peak_managed_bytes, memory_limit)?;
    let mut encoder = encoder.session(destination, &plan.output)?;

    for y in 0..height {
        let pixels = pull(&mut source, &mut stages, &mut scratch, y)?;
        pack(pixels.row(y), plan.output, &mut output_bytes);
        encoder.write_rows(Rows::new(&plan.output, output_bytes.len(), &output_bytes)?)?;
    }

    // A stencil looking only behind the current row may leave trailing input unread.
    source.advance(height - 1)?;
    source.decoder.finish()?;
    encoder.finish()?;
    Ok(stats)
}

fn stats(
    backend: Backend,
    peak_managed_bytes: usize,
    memory_limit: usize,
) -> Result<ExecutionStats> {
    check_budget(peak_managed_bytes, memory_limit)?;
    Ok(ExecutionStats {
        backend,
        peak_managed_bytes,
    })
}

fn buffer<T: Clone>(len: usize, value: T) -> Result<Vec<T>> {
    let mut data = Vec::new();
    data.try_reserve_exact(len)
        .map_err(|_| Error::Limit(format!("could not allocate a {len}-element working buffer")))?;
    data.resize(len, value);
    Ok(data)
}

fn decode_row(decoder: &mut dyn DecoderSession, info: &ImageInfo, bytes: &mut [u8]) -> Result<()> {
    let output = RowsMut::new(info, info.row_bytes(), bytes)?;
    match decoder.read_rows(output)? {
        1 => Ok(()),
        0 => Err(Error::Codec(
            "decoder ended before the declared image height".into(),
        )),
        _ => Err(Error::Codec(
            "decoder returned more rows than requested".into(),
        )),
    }
}

fn unpack(bytes: &[u8], info: ImageInfo, row: &mut [f32]) {
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

fn pack(row: &[f32], info: ImageInfo, bytes: &mut [u8]) {
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

struct Ring {
    info: ImageInfo,
    capacity: usize,
    data: Vec<f32>,
}

impl Ring {
    fn new(info: ImageInfo, capacity: usize) -> Result<Self> {
        Ok(Self {
            info,
            capacity,
            data: buffer(float_ring_bytes(info, capacity)? / size_of::<f32>(), 0.0)?,
        })
    }

    fn capacity_bytes(&self) -> usize {
        self.data.capacity() * size_of::<f32>()
    }

    fn row_range(&self, index: usize) -> std::ops::Range<usize> {
        let len = self.info.row_bytes(); // A packed byte row contains one byte per sample.
        let start = (index % self.capacity) * len;
        start..start + len
    }

    fn row(&self, index: usize) -> &[f32] {
        &self.data[self.row_range(index)]
    }

    fn row_mut(&mut self, index: usize) -> &mut [f32] {
        let range = self.row_range(index);
        &mut self.data[range]
    }
}

struct FloatSource {
    decoder: Box<dyn DecoderSession>,
    pixels: Ring,
    bytes: Vec<u8>,
    next: usize,
}

impl FloatSource {
    fn advance(&mut self, y: usize) -> Result<()> {
        let info = self.pixels.info;
        while self.next <= y {
            decode_row(&mut *self.decoder, &info, &mut self.bytes)?;
            unpack(&self.bytes, info, self.pixels.row_mut(self.next));
            self.next += 1;
        }
        Ok(())
    }
}

struct StageState<'a> {
    stage: &'a Stage,
    analysis: &'a StagePlan,
    pixels: Ring,
    next: usize,
}

struct Scratch {
    scalar: Vec<f32>,
    vector: Vec<f32x4>,
}

// Pull upstream through the highest required row. The inferred span guarantees
// every earlier row needed for this output is still in the upstream ring.
fn pull<'a>(
    source: &'a mut FloatSource,
    stages: &'a mut [StageState<'_>],
    scratch: &mut Scratch,
    target: usize,
) -> Result<&'a Ring> {
    let Some((stage, upstream)) = stages.split_last_mut() else {
        source.advance(target)?;
        return Ok(&source.pixels);
    };

    while stage.next <= target {
        let y = stage.next;
        let height = stage.analysis.input.height() as usize;
        let required = offset(y, stage.analysis.max_y, height);
        let input = pull(source, upstream, scratch, required)?;
        evaluate_row(stage.stage, input, y, stage.pixels.row_mut(y), scratch);
        stage.next += 1;
    }
    Ok(&stage.pixels)
}

#[inline]
fn offset(position: usize, delta: i32, extent: usize) -> usize {
    position
        .saturating_add_signed(delta as isize)
        .min(extent - 1)
}

fn evaluate_row(stage: &Stage, input: &Ring, y: usize, output: &mut [f32], scratch: &mut Scratch) {
    let width = input.info.width() as usize;
    let programs = stage.programs();
    for (channel, plane) in output.chunks_exact_mut(width).enumerate() {
        let program = &programs[if programs.len() == 1 { 0 } else { channel }];
        let mut vector_end = 0;
        if !scratch.vector.is_empty() {
            let (groups, tail) = plane.as_chunks_mut::<4>();
            for (group, pixels) in groups.iter_mut().enumerate() {
                *pixels = evaluate(program, input, (group * 4, y), channel, &mut scratch.vector)
                    .to_array();
            }
            vector_end = width - tail.len();
        }
        for (x, pixel) in plane.iter_mut().enumerate().skip(vector_end) {
            *pixel = evaluate(program, input, (x, y), channel, &mut scratch.scalar);
        }
    }
}

trait Lane:
    Copy
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
{
    fn splat(value: f32) -> Self;
    fn sample(row: &[f32], x: usize, dx: i32) -> Self;
    fn min(self, rhs: Self) -> Self;
    fn max(self, rhs: Self) -> Self;
}

impl Lane for f32 {
    #[inline(always)]
    fn splat(value: f32) -> Self {
        value
    }
    #[inline(always)]
    fn sample(row: &[f32], x: usize, dx: i32) -> Self {
        row[offset(x, dx, row.len())]
    }
    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        f32::min(self, rhs) + 0.0
    }
    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        f32::max(self, rhs) + 0.0
    }
}

impl Lane for f32x4 {
    #[inline(always)]
    fn splat(value: f32) -> Self {
        f32x4::splat(value)
    }
    #[inline(always)]
    fn sample(row: &[f32], x: usize, dx: i32) -> Self {
        // Interior vectors load contiguous samples. At an edge, clamp each lane independently.
        if let Some(start) = x.checked_add_signed(dx as isize)
            && let Some(samples) = row.get(start..start + 4)
        {
            return f32x4::new(samples.try_into().expect("four contiguous samples"));
        }
        f32x4::new(std::array::from_fn(|lane| {
            row[offset(x + lane, dx, row.len())]
        }))
    }
    #[inline(always)]
    fn min(self, rhs: Self) -> Self {
        f32x4::min(self, rhs) + f32x4::ZERO
    }
    #[inline(always)]
    fn max(self, rhs: Self) -> Self {
        f32x4::max(self, rhs) + f32x4::ZERO
    }
}

#[inline]
fn evaluate<T: Lane>(
    program: &Expr,
    input: &Ring,
    (x, y): (usize, usize),
    channel: usize,
    stack: &mut [T],
) -> T {
    let width = input.info.width() as usize;
    let height = input.info.height() as usize;
    let last_channel = input.info.format().channels() - 1;
    let mut depth = 0;
    for &instruction in program.instructions() {
        match instruction {
            Instruction::Constant(value) => {
                stack[depth] = T::splat(value);
                depth += 1;
            }
            Instruction::Sample {
                dx,
                dy,
                channel: selected,
            } => {
                let row = input.row(offset(y, dy, height));
                let channel = selected.unwrap_or(channel).min(last_channel);
                stack[depth] = T::sample(&row[channel * width..][..width], x, dx);
                depth += 1;
            }
            operator => {
                let lhs = stack[depth - 2];
                let rhs = stack[depth - 1];
                depth -= 1;
                stack[depth - 1] = match operator {
                    Instruction::Add => lhs + rhs,
                    Instruction::Sub => lhs - rhs,
                    Instruction::Mul => lhs * rhs,
                    Instruction::Div => lhs / rhs,
                    Instruction::Min => lhs.min(rhs),
                    Instruction::Max => lhs.max(rhs),
                    _ => unreachable!("non-binary instruction"),
                };
            }
        }
    }
    stack[0]
}
