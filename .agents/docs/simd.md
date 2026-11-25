# SIMD

The same postfix evaluator is specialized for scalar `f32` and four-lane `wide::f32x4`. It evaluates adjacent x coordinates in one channel together. Planar float rows give contiguous loads for horizontal offsets and selected channels. Vertical offsets choose the corresponding row ring slot. Border vectors clamp each lane independently. The final one to three pixels use the scalar evaluator.

`Backend::Auto` selects the SIMD evaluator when the build targets SSE2 on x86 or NEON on AArch64 and runtime detection confirms the feature. `Backend::Simd` also falls back to scalar if the SIMD evaluator is unavailable. `Backend::Scalar` runs the reference evaluator. Inspect `image.explain()?.backend` to identify the resolved backend. Byte normalization and final packing use shared code.

The project evaluated Rust's experimental portable SIMD and `wide`. The latter supports stable Rust and generates native arithmetic for both baseline targets without maintaining handwritten intrinsics. [wide source](https://github.com/Lokathor/wide)

Exact final byte equivalence is tested on deterministic inputs. Floating-point NaN payloads and the sign of zero are not part of the output contract. Use the [benchmark harness](../../benchmarks/README.md) for host-specific throughput comparisons.
