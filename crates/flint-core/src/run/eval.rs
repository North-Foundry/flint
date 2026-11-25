use super::{buffer, offset, rows::Ring};
use crate::Backend;
use flint_sdk::{
    Expr, Result,
    graph::{Instruction, Stage},
};
use wide::f32x4;

pub(super) struct Scratch {
    scalar: Vec<f32>,
    vector: Vec<f32x4>,
}

impl Scratch {
    pub(super) fn new(depth: usize, backend: Backend) -> Result<Self> {
        Ok(Self {
            scalar: buffer(depth, 0.0)?,
            vector: if backend == Backend::Simd {
                buffer(depth, f32x4::ZERO)?
            } else {
                Vec::new()
            },
        })
    }

    pub(super) fn capacity_bytes(&self) -> usize {
        self.scalar.capacity() * size_of::<f32>() + self.vector.capacity() * size_of::<f32x4>()
    }
}

pub(super) fn evaluate_row(
    stage: &Stage,
    input: &Ring,
    y: usize,
    output: &mut [f32],
    scratch: &mut Scratch,
) {
    let width = input.info().width() as usize;
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
    let info = input.info();
    let width = info.width() as usize;
    let height = info.height() as usize;
    let last_channel = info.format().channels() - 1;
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
