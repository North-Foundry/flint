# Testing

Run the workspace checks before reporting changes.

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-features --locked
cargo test -p flint --no-default-features --locked
python3 benchmarks/run.py --output-dir benchmarks/output --iterations 3
```

Tests in `tests/pipeline.rs` are registered under the facade package. Their custom codec provides controlled samples and observes lazy execution without touching core internals. Use independent reference calculations for operations and widths that exercise scalar tails. Codec packages cover direct extension contracts with nonseekable streams, incomplete data, and output failures.

Compare `Optimization::Auto` and `Optimization::Disabled` with the same graph
and requested backend. Check the selected backend, kernel ranges, folded
instruction counts, output bytes, and the plan estimate against measured buffer
capacity. Exercise `f32` edge cases before output quantization and include
widths with full SIMD groups and each scalar tail length.

Benchmarks separate process-level conversion from in-process evaluator
comparisons. Single-stage pointwise and stencil workloads measure folding in
isolation. A chain workload includes fusion, and a pointwise control has no
constant arithmetic. Record planning time, elapsed time, throughput, source
and executable instruction counts, folds, scratch, estimated and measured
managed bytes, and peak RSS as distinct metrics. Compare output checksums for
each pair. Missing external tools are reported by the harness.
