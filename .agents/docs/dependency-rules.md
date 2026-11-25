# Dependency rules

- `flint-sdk` has no third-party dependencies. It owns the public Rust extension contracts.
- `flint-core` depends on the SDK and uses `wide` for vector arithmetic. It never imports an official codec or operation.
- `flint-png`, `flint-jpeg`, and `flint-basic` depend on the SDK. Codecs can also use format-specific libraries. External extensions follow the same rules.
- `flint` depends on the core and SDK. Its `png` and `jpeg` features independently include the official codec crates. It uses `tempfile` to stage file output for atomic replacement on success.
- Consumers choose their operation dependencies. `flint-basic` is a development dependency of the facade for examples and tests.
