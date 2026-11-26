# Flint project instructions

This is the single project instruction entry point. Read the applicable
guides in `.agents/docs/` before editing.

| Guide | When to use it |
| --- | --- |
| [Architecture](.agents/docs/architecture.md) | Understand the workspace, crate boundaries, and processing pipeline. |
| [Dependency rules](.agents/docs/dependency-rules.md) | Change crate boundaries or dependencies. |
| [Design principles](.agents/docs/principles.md) | Evaluate design decisions against Flint's core contracts. |
| [Code style](.agents/docs/code-style.md) | Write or modify Rust code. |
| [Symbolic primitives](.agents/docs/primitives.md) | Work with the symbolic building blocks used in expressions. |
| [Expressions](.agents/docs/expressions.md) | Change expression construction or execution. |
| [Symbolic optimization](.agents/docs/optimizer.md) | Change constant folding, executable kernels, fusion, or profitability policy. |
| [Streaming](.agents/docs/streaming.md) | Change execution while preserving row streaming and graph laziness. |
| [SIMD](.agents/docs/simd.md) | Change vectorized execution while preserving numeric behavior. |
| [Writing an operation](.agents/docs/writing-an-operation.md) | Add or register an operation extension. |
| [Writing a decoder](.agents/docs/writing-a-decoder.md) | Add or register a decoder extension. |
| [Writing an encoder](.agents/docs/writing-an-encoder.md) | Add or register an encoder extension. |
| [Testing](.agents/docs/testing.md) | Run formatting, Clippy, workspace tests, and benchmarks. |
| [Commits](.agents/docs/commits.md) | Write commit messages using the repository convention. |

Keep this index in sync with the guides when changing `.agents/docs/`.

## Model routing

Delegate useful independent tasks with narrow ownership and evidence requirements. Keep small tasks local. Select `gpt-6-luna` at medium for focused exploration and mechanical work, `gpt-6-luna` at high for localized implementation, and the available Sol model at high for architectural work and correctness reviews. The main agent integrates changes and verifies the result.
