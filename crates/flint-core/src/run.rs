use self::{
    evaluation::{Scratch, evaluate_row},
    rows::{Ring, pack, unpack},
};
use crate::{
    Backend, ExecutionPlan, StagePlan,
    codec::{DecoderSession, SelectedDecoder, SelectedEncoder},
    plan::{check_budget, graph_stages},
};
use flint_sdk::{
    Destination, Error, ImageExpression, ImageInformation, Result, Rows, RowsMut, Source,
    graph::Stage,
};

mod evaluation;
mod rows;

/**
 * The selected backend and measured working memory from a completed execution.
 */
#[derive(Clone, Copy, Debug)]
pub struct ExecutionStatistics {
    pub backend: Backend,
    /// Peak total capacity of managed pixel and evaluator buffers, in bytes.
    pub peak_managed_bytes: usize,
}

/**
 * Run the graph row by row within the managed memory limit.
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
    let mut scratch = Scratch::new(stack_depth, plan.backend)?;
    // These buffers coexist throughout processing. Their capacities never change.
    let peak_managed_bytes = source.bytes.capacity()
        + output_bytes.capacity()
        + source.pixels.capacity_bytes()
        + stages
            .iter()
            .map(|stage| stage.pixels.capacity_bytes())
            .sum::<usize>()
        + scratch.capacity_bytes();
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

/**
 * Validate measured buffer capacity and record the execution statistics.
 */
fn stats(
    backend: Backend,
    peak_managed_bytes: usize,
    memory_limit: usize,
) -> Result<ExecutionStatistics> {
    check_budget(peak_managed_bytes, memory_limit)?;
    Ok(ExecutionStatistics {
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

struct StageState<'a> {
    stage: &'a Stage,
    analysis: &'a StagePlan,
    pixels: Ring,
    next: usize,
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
