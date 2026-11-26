use self::{
    evaluation::{Scratch, evaluate_pointwise_row, evaluate_row},
    rows::{Ring, pack, unpack},
};
use crate::{
    Backend, ExecutionPlan, KernelPlan, Optimization,
    codec::{DecoderSession, SelectedDecoder, SelectedEncoder},
    plan::{CompiledPlan, KernelProgram, add, check_budget},
};
use flint_sdk::{
    Destination, Error, ImageExpression, ImageInformation, Result, Rows, RowsMut, Source,
};

mod evaluation;
mod rows;

/**
 * The kernels, selected backend, and measured memory of a completed execution.
 */
#[derive(Clone, Copy, Debug)]
pub struct ExecutionStatistics {
    pub backend: Backend,
    /** Number of stages in the symbolic graph. */
    pub original_stage_count: usize,
    /** Number of executable kernels after planning. */
    pub kernel_count: usize,
    /// Peak total capacity of managed pixel and evaluator buffers, in bytes.
    pub peak_managed_bytes: usize,
}

/**
 * Settings needed by the core planner and streaming runtime.
 */
#[derive(Clone, Copy, Debug)]
pub struct ExecutionOptions {
    pub backend: Backend,
    pub optimization: Optimization,
    pub memory_limit: usize,
}

/**
 * Run the graph row by row with automatic optimization and the given limit.
 */
pub fn execute(
    image: &ImageExpression,
    backend: Backend,
    memory_limit: usize,
    source: &dyn Fn() -> Result<Source>,
    decoder: SelectedDecoder,
    encoder: SelectedEncoder,
    destination: Destination,
) -> Result<ExecutionStatistics> {
    execute_with_options(
        image,
        ExecutionOptions {
            backend,
            optimization: Optimization::Auto,
            memory_limit,
        },
        source,
        decoder,
        encoder,
        destination,
    )
}

/**
 * Select executable kernels and run them row by row within the managed limit.
 * Returns a resource-limit error if the runtime's buffers cannot fit.
 */
pub fn execute_with_options(
    image: &ImageExpression,
    options: ExecutionOptions,
    source: &dyn Fn() -> Result<Source>,
    decoder: SelectedDecoder,
    encoder: SelectedEncoder,
    destination: Destination,
) -> Result<ExecutionStatistics> {
    let memory_limit = options.memory_limit;
    let compiled = CompiledPlan::prepare(
        image,
        decoder.name(),
        options.backend,
        options.optimization,
        memory_limit,
    )?;
    let plan = &compiled.report;
    check_budget(plan.managed_bytes, memory_limit)?;
    let mut decoder = decoder.session(source()?)?;
    if *decoder.info() != plan.source {
        return Err(Error::Invalid(
            "source metadata changed since the image was opened".into(),
        ));
    }
    let mut input_bytes = buffer(plan.source.row_bytes(), 0u8)?;

    let height = plan.source.height() as usize;
    if plan.kernels.is_empty() {
        let stats = stats(plan, input_bytes.capacity(), memory_limit)?;
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
        pixels: Ring::new(plan.source, plan.kernels[0].buffered_rows)?,
        bytes: input_bytes,
        next: 0,
    };
    let mut output_bytes = buffer(plan.output.row_bytes(), 0u8)?;
    let mut kernels = compiled
        .programs
        .iter()
        .zip(&plan.kernels)
        .enumerate()
        .map(|(index, (program, analysis))| {
            let rows = plan
                .kernels
                .get(index + 1)
                .map_or(1, |next| next.buffered_rows);
            Ok(KernelState {
                program,
                analysis,
                pixels: Ring::new(analysis.output, rows)?,
                next: 0,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut scratch = Scratch::new(
        plan.scratch_slots(),
        plan.backend.uses_vectors(plan.source.width()),
    )?;
    // These buffers coexist throughout processing. Their capacities never change.
    let peak_managed_bytes = [
        source.bytes.capacity(),
        output_bytes.capacity(),
        source.pixels.capacity_bytes(),
        scratch.capacity_bytes(),
    ]
    .into_iter()
    .chain(kernels.iter().map(|kernel| kernel.pixels.capacity_bytes()))
    .try_fold(0, add)?;
    let stats = stats(plan, peak_managed_bytes, memory_limit)?;
    let mut encoder = encoder.session(destination, &plan.output)?;

    for y in 0..height {
        let pixels = pull(&mut source, &mut kernels, &mut scratch, y)?;
        pack(pixels.row(y), plan.output, &mut output_bytes);
        encoder.write_rows(Rows::new(&plan.output, output_bytes.len(), &output_bytes)?)?;
    }

    // A stencil looking only behind the current row may leave trailing input unread.
    source.advance(height - 1)?;
    source.decoder.finish()?;
    encoder.finish()?;
    Ok(stats)
}

/**
 * Validate measured buffer capacity and record the execution statistics.
 */
fn stats(
    plan: &ExecutionPlan,
    peak_managed_bytes: usize,
    memory_limit: usize,
) -> Result<ExecutionStatistics> {
    check_budget(peak_managed_bytes, memory_limit)?;
    Ok(ExecutionStatistics {
        backend: plan.backend,
        original_stage_count: plan.stages.len(),
        kernel_count: plan.kernels.len(),
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

/**
 * Read exactly one packed row and validate the decoder's reported count.
 */
fn decode_row(
    decoder: &mut dyn DecoderSession,
    info: &ImageInformation,
    bytes: &mut [u8],
) -> Result<()> {
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

struct FloatSource {
    decoder: Box<dyn DecoderSession>,
    pixels: Ring,
    bytes: Vec<u8>,
    next: usize,
}

impl FloatSource {
    fn advance(&mut self, y: usize) -> Result<()> {
        let info = self.pixels.info();
        while self.next <= y {
            decode_row(&mut *self.decoder, &info, &mut self.bytes)?;
            unpack(&self.bytes, info, self.pixels.row_mut(self.next));
            self.next += 1;
        }
        Ok(())
    }
}

/**
 * A kernel's immutable program and the ring window produced by this execution.
 */
struct KernelState<'plan, 'graph> {
    program: &'plan KernelProgram<'graph>,
    analysis: &'plan KernelPlan,
    pixels: Ring,
    next: usize,
}

/**
 * Pull upstream through the highest required row. The inferred span guarantees
 * every earlier row needed for this output is still in the upstream ring.
 */
fn pull<'state>(
    source: &'state mut FloatSource,
    kernels: &'state mut [KernelState<'_, '_>],
    scratch: &mut Scratch,
    target: usize,
) -> Result<&'state Ring> {
    let Some((kernel, upstream)) = kernels.split_last_mut() else {
        source.advance(target)?;
        return Ok(&source.pixels);
    };

    while kernel.next <= target {
        let y = kernel.next;
        let height = kernel.analysis.input.height() as usize;
        let required = offset(y, kernel.analysis.max_y, height);
        let input = pull(source, upstream, scratch, required)?;
        let output = kernel.pixels.row_mut(y);
        match kernel.program {
            KernelProgram::Original(stage) => evaluate_row(stage, input, y, output, scratch),
            KernelProgram::Pointwise(program) => {
                evaluate_pointwise_row(program, input, y, output, scratch)
            }
        }
        kernel.next += 1;
    }
    Ok(&kernel.pixels)
}

#[inline]
fn offset(position: usize, delta: i32, extent: usize) -> usize {
    position
        .saturating_add_signed(delta as isize)
        .min(extent - 1)
}
