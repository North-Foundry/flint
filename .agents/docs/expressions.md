# Expressions

`Expr` contains a bounded postfix program of constants, sample references, and arithmetic. Use Rust arithmetic operators, `min`, `max`, `clamp`, and `Iterator::sum::<Expr>()`. An empty sum yields zero. The planner infers the maximum operand stack depth from the executable postfix instructions. The evaluator reuses that stack across pixels.

Byte samples are normalized as `f32::from(sample) / 255.0`. Every stage operates on `f32` values in expression order. Intermediate values keep their precision and range. The runtime uses no fused operation when the expression specifies separate multiplication and addition.

Division uses floating-point semantics. Min and max choose the numeric operand if exactly one operand is NaN. Clamp applies max with the lower bound, then min with the upper bound. Final packing clamps into the unit interval, multiplies by 255, and rounds to nearest with ties up. NaN becomes 0, negative infinity becomes 0, and positive infinity becomes 255.

The SDK accepts at most 65,536 instructions in a program and 128 stages per graph. The runtime plan accounts for evaluator workspace. Symbolic graph construction and its memory are separate from the engine budget.

Write a symbolic constant expression as `Expr::from(0.2) + 0.3`. Rust evaluates
`0.2 + 0.3` directly when both operands are native floats. In `Optimization::Auto`,
Flint automatically folds safe symbolic constant arithmetic in standalone
programs and can propagate results across fused pointwise stages. Plugin code
can use expressions as usual. The planner keeps sample-dependent operations such
as `sample * 1.0` and preserves original `f32` operand order. See
[symbolic optimization](optimizer.md).
