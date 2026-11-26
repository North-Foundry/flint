# Kernel fusion

Flint plans kernels from the immutable symbolic graph before it opens a decoder
for execution. An original stage runs its SDK postfix program. A fused kernel
can execute a sequence of pointwise stages in the same streaming row traversal.
`Engine::builder().optimization(Optimization::Disabled)` selects the original
stages for comparison. Automatic fusion and backend selection are independent.

Eligibility comes from the sample offsets in every output program. A constant
program and a stencil that only samples `(0, 0)` are pointwise. Any other
offset ends a fusion group. Stage formats and channel selection follow the
SDK's rules, including broadcasting Gray8 samples in RGB expressions.

During planning, each postfix program becomes a sequence of ordered value
definitions. Downstream samples reference the preceding stage's channel values
directly. Identical definitions share one value, using constant bits and ordered
operands as the key. Only definitions reachable from the final channels are
lowered. Each live definition is evaluated once per pixel or SIMD group. A
temporary slot becomes reusable after its last consumer, and final values stay
live until the kernel writes its output channels.

Arithmetic remains in `f32` and uses the existing scalar and SIMD lane
operations. The optimizer retains intermediate clamps and the original operand
order. It applies no floating-point algebraic simplifications. SIMD evaluates
four adjacent pixels in each group and keeps scalar evaluation for the tail.

The planner caps each candidate at 2,048 source instructions across output
channels and 64 live value slots. This admits a 50-stage RGB brightness chain
and splits longer arithmetic-intensive chains. Even a full SIMD workspace
uses at most 1,280 bytes for fused values, including the scalar tail. The
planner compares executable instruction count
and additional scratch against the source programs and eliminated rows. A
rejected candidate can be retried with a shorter prefix or left as original
stages. The planner sizes rings from the chosen kernels and their consumers'
spatial spans. It considers the original plan if the optimized estimate exceeds
the budget.

`Image::explain()` reports original stages, executable kernel ranges, sample
spans, instruction counts, temporary slots, and estimated managed memory.
Execution statistics report counts and measured managed capacity. The scalar
and SIMD scratch allocations serve both postfix operands and fused values.
Rows narrower than four pixels require scalar scratch only.
Scratch and pixel rows count toward the managed memory limit. Graph metadata,
compiled instructions, planning allocations, runtime row bookkeeping, allocator
overhead, and codec state are outside the limit. Plan storage is bounded by the SDK graph limits
and optimizer caps and does not scale with image dimensions.

The optimizer preserves planar row storage, spatial boundaries and direct
packed-row transcoding. Operation developers describe their computation through
the SDK. Immutable kernel programs and their dependency metadata can be shared
with future workers, while row rings and evaluator scratch belong to each
execution.

[Benchmark methodology](../../benchmarks/README.md)
