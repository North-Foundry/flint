# Symbolic optimization

Flint plans kernels from the immutable symbolic graph before it opens a decoder
for execution. `Optimization::Auto` enables safe constant folding and profitable
pointwise fusion. `Engine::builder().optimization(Optimization::Disabled)`
executes the original stages and instructions for comparison. Backend selection
is independent of the optimization setting.

## Constant folding

The planner evaluates addition, subtraction, multiplication, division, minimum,
and maximum when both operands are constants. Clamp is represented as maximum
followed by minimum and can fold at either step. Newly folded results can become
operands of subsequent folds. The planner evaluates each operation in order with
the evaluator's `f32` arithmetic, including positive-zero normalization for
min and max. A NaN result or a bit disagreement between scalar and active SIMD
evaluators leaves the arithmetic in the executable program. The optimizer
preserves all sample-dependent arithmetic and the order of floating-point operations.

Standalone pointwise and spatial stages share the same postfix interpreter
logic. The runtime selects a specialized row loop from the planned program.
Unchanged kernels use their original SDK programs, and folded kernels use
compacted instructions. Folding preserves sample offsets and channel selection for
horizontal, vertical, and general stencils. A kernel boundary ends constant
propagation. Source decoding, metadata validation, and codec finalization also
run for constant output.

## Pointwise fusion

Eligibility comes from the sample offsets in every output program. A constant
program and a stencil that only samples `(0, 0)` are pointwise. Any other
offset ends a fusion group. Stage formats and channel selection follow the
SDK's rules, including broadcasting Gray8 samples in RGB expressions.

During planning, each postfix program becomes a sequence of ordered value
definitions. Downstream samples reference the preceding stage's channel values
directly, allowing constants to propagate across compatible stages and channels.
The planner folds constant arithmetic before interning each definition. Identical
definitions share one value, using constant bits and ordered operands as the
key. Only definitions reachable from the final channels are
lowered. Each live definition is evaluated once per pixel or SIMD group. A
temporary slot becomes reusable after its last consumer, and final values stay
live until the kernel writes its output channels.

The value evaluator uses the same lane operations and preserves operand order.
SIMD evaluates four adjacent pixels per group and uses scalar evaluation for
the tail.

The planner caps each candidate at 2,048 source instructions across output
channels and 64 live value slots. This admits a 50-stage RGB brightness chain
and splits longer arithmetic-intensive chains. Even a full SIMD workspace
uses at most 1,280 bytes for fused values, including the scalar tail. The
planner compares the lowered instruction count with the source count and checks
whether eliminated rows pay for any extra scratch over standalone execution.
It computes folded postfix stack depths when they can affect that decision. A
rejected candidate can be retried with a shorter prefix or left as standalone
stages, where postfix folding can still apply. The planner sizes rings from the
chosen kernels and their consumers' spatial spans. It considers standalone
kernels if the fused estimate exceeds the budget.

## Accounting and diagnostics

`Image::explain()` reports source and executable instruction counts, kernel
ranges, sample spans, scratch requirements, and estimated managed memory.
The fold count records successful source arithmetic occurrences in selected
kernels, expanded per output channel and counted before common subexpression
elimination and dead definition removal. It can include arithmetic in a fused
stage whose result is later dropped. Execution statistics also report the fold
count and measured managed capacity. Scratch is sized from the largest executable
postfix stack or fused value workspace. Rows narrower than four pixels require
scalar scratch only.
Scratch and pixel rows count toward the managed memory limit. Graph metadata,
compiled instructions, planning allocations, runtime row bookkeeping, allocator
overhead, and codec state are outside the limit. Plan storage is bounded by the SDK graph limits
and optimizer caps and does not scale with image dimensions.

The optimizer preserves planar row storage, spatial boundaries and direct
packed-row transcoding. Operation developers describe their computation through
the SDK and get constant folding automatically. Immutable kernel programs and
their dependency metadata can be shared
with future workers, while row rings and evaluator scratch belong to each
execution.

[Benchmark methodology](../../benchmarks/README.md)
