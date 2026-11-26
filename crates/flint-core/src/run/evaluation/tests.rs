/*!
 * Compare lowered pointwise kernels against materialized postfix stages before packing.
 */

use super::{
    Ring, Scratch, evaluate_folded_row, evaluate_original_row, evaluate_pointwise_row,
    fold_constants,
};
use crate::{
    Backend, KernelPlan, Optimization,
    plan::{CompiledPlan, KernelProgram},
};
use flint_sdk::{
    Expr, ImageExpression, ImageInformation, Pixel, PixelFormat, Result, graph::Instruction,
};
use std::hint::black_box;

/**
 * Fold each binary operator with the same exact result in scalar and SIMD
 * plans, including rounding, gradual underflow, and both zero signs.
 */
#[test]
fn constant_folding_preserves_result_bits() {
    use Instruction::{Add, Div, Max, Min, Mul, Sub};

    let subnormal = f32::from_bits(1);
    for (operator, left, right, expected) in [
        (Add, 2.0, 3.0, 5.0),
        (Add, 1.0, f32::EPSILON / 2.0, 1.0),
        (Add, f32::MAX, f32::MAX, f32::INFINITY),
        (Sub, 2.0, 3.0, -1.0),
        (Mul, 2.0, 3.0, 6.0),
        (Mul, 3.0, -0.0, -0.0),
        (Div, 1.0, 10.0, f32::from_bits(0x3dcc_cccd)),
        (Div, 1.0, -0.0, f32::NEG_INFINITY),
        (Div, subnormal, subnormal, 1.0),
        (Div, f32::MIN_POSITIVE, 2.0, f32::from_bits(0x0040_0000)),
        (Div, -subnormal, 2.0, -0.0),
        (Min, -0.0, -0.0, 0.0),
        (Max, 0.0, -0.0, 0.0),
        (Min, f32::NAN, 0.75, 0.75),
        (Max, 0.25, f32::NAN, 0.25),
        (Max, 0.0, subnormal, subnormal),
    ] {
        for use_vectors in [false, true] {
            let (operator, left, right) = black_box((operator, left, right));
            assert_eq!(
                fold_constants(operator, left, right, use_vectors).map(f32::to_bits),
                Some(expected.to_bits()),
                "{operator:?} with {left:?} and {right:?}, use_vectors={use_vectors}"
            );
        }
    }
}

/**
 * Leave non-binary instructions and operations producing NaN for execution.
 */
#[test]
fn constant_folding_rejects_non_binary_instructions_and_nan_results() {
    for use_vectors in [false, true] {
        for operator in [
            Instruction::Constant(2.0),
            Instruction::Sample {
                dx: 0,
                dy: 0,
                channel: None,
            },
            Instruction::Div,
        ] {
            assert_eq!(fold_constants(operator, 0.0, -0.0, use_vectors), None);
        }
        assert_eq!(
            fold_constants(Instruction::Min, f32::NAN, f32::NAN, use_vectors),
            None
        );
    }
}

/**
 * A wholly constant fused computation still rounds at each original operation
 * and preserves the sign of a zero that crosses a channel boundary.
 */
#[test]
fn fused_constant_channels_keep_intermediate_rounding_and_signed_zero() -> Result<()> {
    let image = ImageExpression::source(ImageInformation::new(7, 1, PixelFormat::Gray8)?)
        .map_pixel(|_| Pixel::rgb(Expr::from(16_777_216.0) + 1.0, Expr::from(-0.0) * 2.0, 0.25))?
        .map_pixel(|pixel| {
            Pixel::rgb(
                pixel.r - 16_777_216.0,
                1.0 / pixel.g,
                pixel.b.clamp(0.0, 1.0),
            )
        })?;
    assert_fused_rows_match(&image, &[&[0.0; 7]], |kernel| {
        assert_eq!(kernel.folded_instructions, 6);
        assert_eq!(kernel.instruction_count, 3);
        assert_eq!(kernel.temporary_values, 3);
    })
}

/**
 * Compare standalone postfix outputs before quantization, including clamping,
 * the sign of an underflowed result, and an arithmetic NaN kept at runtime.
 */
#[test]
fn standalone_constants_match_unmodified_postfix_bits() -> Result<()> {
    let information = ImageInformation::new(7, 1, PixelFormat::Rgb8)?;
    let image = ImageExpression::source(information).map_pixel(|_| {
        Pixel::rgb(
            (Expr::from(f32::MIN_POSITIVE) / 2.0).clamp(0.0, 1.0),
            Expr::from(-f32::from_bits(1)) / 2.0,
            Expr::from(f32::INFINITY) - f32::INFINITY,
        )
    })?;
    let ring = Ring::new(information, 1)?;
    for backend in [Backend::Scalar, Backend::Simd] {
        let mut rows = [[0.0; 21]; 2];
        for (optimization, row) in [Optimization::Disabled, Optimization::Auto]
            .into_iter()
            .zip(&mut rows)
        {
            let plan = CompiledPlan::prepare(&image, "raw", backend, optimization, usize::MAX)?;
            let mut scratch = Scratch::new(plan.report.scratch_slots(), backend.uses_vectors(7))?;
            match (optimization, plan.programs.as_slice()) {
                (Optimization::Disabled, [KernelProgram::Original(stage)]) => {
                    evaluate_original_row(stage, &ring, 0, row, &mut scratch);
                }
                (Optimization::Auto, [KernelProgram::FoldedPostfix(programs)]) => {
                    evaluate_folded_row(programs, &ring, 0, row, &mut scratch);
                }
                _ => panic!("expected matching standalone program"),
            }
        }
        for (original, optimized) in rows[0].iter().zip(rows[1]) {
            if original.is_nan() {
                assert!(optimized.is_nan());
            } else {
                assert_eq!(original.to_bits(), optimized.to_bits());
            }
        }
        assert_eq!(rows[1][0].to_bits(), 0x0040_0000);
        assert_eq!(rows[1][7].to_bits(), (-0.0f32).to_bits());
        assert!(rows[1][14].is_nan());
    }
    Ok(())
}

/**
 * Compare exact output bits on both evaluator paths using the same raw planar rows.
 * Select the lane type directly so the wide evaluator is exercised on every host.
 * Each execution reuses its row storage and scratch across input rows.
 */
fn assert_fused_rows_match(
    image: &ImageExpression,
    input_rows: &[&[f32]],
    check_kernel: impl Fn(&KernelPlan),
) -> Result<()> {
    for backend in [Backend::Scalar, Backend::Simd] {
        let original =
            CompiledPlan::prepare(image, "raw", backend, Optimization::Disabled, usize::MAX)?;
        let fused = CompiledPlan::prepare(image, "raw", backend, Optimization::Auto, usize::MAX)?;
        let [KernelProgram::Pointwise(program)] = fused.programs.as_slice() else {
            panic!("expected one fused kernel on {backend:?}");
        };
        check_kernel(&fused.report.kernels[0]);

        let mut rings = std::iter::once(original.report.source)
            .chain(original.report.kernels.iter().map(|kernel| kernel.output))
            .map(|information| Ring::new(information, 1))
            .collect::<Result<Vec<_>>>()?;
        let use_vectors = backend.uses_vectors(original.report.source.width());
        let mut original_scratch = Scratch::new(original.report.scratch_slots(), use_vectors)?;
        let mut fused_scratch = Scratch::new(fused.report.scratch_slots(), use_vectors)?;
        let output_information = fused.report.output;
        let width = output_information.width() as usize;
        let mut fused_row = vec![0.0; output_information.row_bytes()];
        for (row, samples) in input_rows.iter().enumerate() {
            rings[0].row_mut(row).copy_from_slice(samples);
            for (index, program) in original.programs.iter().enumerate() {
                let KernelProgram::Original(stage) = program else {
                    panic!("expected an original stage with optimization disabled");
                };
                let (inputs, outputs) = rings.split_at_mut(index + 1);
                evaluate_original_row(
                    stage,
                    &inputs[index],
                    row,
                    outputs[0].row_mut(row),
                    &mut original_scratch,
                );
            }
            fused_row.fill(f32::from_bits(0x7fc0_00de));
            evaluate_pointwise_row(program, &rings[0], row, &mut fused_row, &mut fused_scratch);
            let original_row = rings.last().unwrap().row(row);
            assert_eq!(fused_row.len(), original_row.len());
            for (sample, (&fused_value, &original_value)) in
                fused_row.iter().zip(original_row).enumerate()
            {
                let channel = sample / width;
                let column = sample % width;
                assert_eq!(
                    fused_value.to_bits(),
                    original_value.to_bits(),
                    "{backend:?}, width {width}, row {row}, channel {channel}, column {column}"
                );
            }
        }
    }
    Ok(())
}

/**
 * Preserve both zero signs and NaN payload bits when a zero-offset stencil
 * forwards samples to aliased Gray-to-RGB outputs. Exercise every SIMD tail length.
 */
#[test]
fn fused_forwarding_preserves_raw_sample_bits() -> Result<()> {
    let samples = [
        -0.0,
        0.0,
        f32::from_bits(0x7fc0_0123),
        f32::INFINITY,
        f32::from_bits(0xffc0_0456),
        f32::NEG_INFINITY,
        f32::from_bits(1),
    ];
    for width in [1, 4, 5, 6, 7] {
        let information = ImageInformation::new(width, 1, PixelFormat::Gray8)?;
        let image = ImageExpression::source(information)
            .stencil(|neighborhood| neighborhood.at(0, 0))?
            .map_pixel(|pixel| Pixel::rgb(pixel.b, pixel.g, pixel.r))?;
        assert_fused_rows_match(&image, &[&samples[..width as usize]], |_| {})?;
    }
    Ok(())
}

/**
 * Distinguish signed-zero constants through reciprocals and observe whether
 * symbolic min, max, and clamp choose numbers when either operand is NaN.
 */
#[test]
fn fused_arithmetic_preserves_signed_zero_and_nan_selection() -> Result<()> {
    let information = ImageInformation::new(9, 1, PixelFormat::Gray8)?;
    let image = ImageExpression::source(information)
        .map_pixel(|pixel| Pixel::rgb(pixel.r * -0.0, pixel.g * 0.0, pixel.b / 0.0))?
        .map_pixel(|pixel| {
            Pixel::rgb(
                1.0 / pixel.r,
                (1.0 / pixel.g).min(f32::NAN),
                pixel.b.clamp(0.25, 0.75),
            )
        })?
        .map_pixel(|pixel| Pixel::rgb(pixel.r, pixel.g, pixel.b - 0.5))?;
    let samples = [
        0.0,
        2.0,
        -1.0,
        -0.0,
        0.5,
        -0.5,
        f32::MIN_POSITIVE,
        -2.0,
        0.0,
    ];
    assert_fused_rows_match(&image, &[&samples], |_| {})
}

/**
 * Repeated Gray inputs and identical RGB expressions share values through a
 * swizzle that skips one intermediate channel and aliases two final outputs.
 */
#[test]
fn fused_channel_changes_share_values_and_reuse_slots() -> Result<()> {
    let information = ImageInformation::new(7, 2, PixelFormat::Rgb8)?;
    let image = ImageExpression::source(information)
        .map_pixel(|pixel| Pixel::gray((pixel.r + pixel.g * 3.0) - pixel.b / 7.0))?
        .map_pixel(|pixel| {
            let shared = pixel.r * pixel.g + pixel.b;
            Pixel::rgb(
                shared.clone() - 3.0,
                shared.clone() * shared.clone(),
                shared,
            )
        })?
        .map_color(|channel| channel + 0.125)?
        .map_pixel(|pixel| Pixel::rgb(pixel.b.clone(), pixel.g, pixel.b))?;
    let mut first = [
        0.25, -0.125, 0.0, 1.0, -0.75, 0.75, 2.0, 1.5, 0.2, 0.7, -0.45, 1.0, -2.0, 0.0, 1.75, -1.5,
        0.4, 1.75, 0.1, 1.25, -4.0,
    ];
    let second = first;
    first.reverse();
    assert_fused_rows_match(&image, &[&first, &second], |kernel| {
        // Three loads, three constants, and nine arithmetic operations suffice.
        assert!(kernel.instruction_count <= 15);
        assert!(kernel.temporary_values < kernel.instruction_count);
    })
}
