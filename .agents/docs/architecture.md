# Architecture

The `flint` facade opens files and source factories, owns engine settings, and enables the default codecs. `flint-sdk` defines the graph, symbolic expressions, and generic row I/O. `flint-core` analyzes and runs graphs and holds the codec factory registry. The plugin crates implement codecs and algorithms through the SDK.

## Crate responsibilities

| Crate | Responsibility |
| --- | --- |
| `flint` | Public facade and consumer API |
| `flint-core` | Planning, optimization, and execution runtime |
| `flint-sdk` | Public extension contracts and symbolic primitives |
| `flint-imaging` | Official collection of image-processing algorithms |
| `flint-png` | PNG codec |
| `flint-jpeg` | JPEG codec |

A symbolic image has a source node followed by immutable operation nodes with references to their inputs. Clones share the graph. Each operation can append stages, and a stage can change the number of channels. Source dimensions remain fixed. The initial runtime follows a single chain. Future multi-input nodes can extend the graph representation alongside their scheduling rules.

A conversion has three phases.

1. Opening reads a short prefix, picks the most recent matching decoder, and opens its session for metadata.
2. Applying operations constructs the graph. Planning resolves a backend, folds safe constant arithmetic in executable programs, selects fusion groups, and sizes storage from the resulting kernels and spatial dependencies. Explaining uses the same planner.
3. Saving or writing plans the kernels and opens a fresh decoder and the selected encoder. The runtime pulls rows through the kernels and finishes both codec sessions.

See [symbolic optimization](optimizer.md) for constant folding, expression composition, and kernel selection.

A future dynamic loader can register factories backed by a versioned C ABI. Keep Rust traits, allocation ownership, and object layouts on the Rust side of that adapter. The current registry contains Rust factory functions for statically linked plugins.
