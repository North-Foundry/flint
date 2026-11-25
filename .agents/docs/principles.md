# Design principles

- Operations describe computation with the SDK. The runtime owns iteration, buffer allocation, sampling, and backend selection.
- Plan before processing. Derive working storage from image width and spatial dependencies so a taller image streams with the same buffers.
- Borrow buffers at codec boundaries. If a conversion has no operations, reuse the same packed row for the decoder and encoder.
- Discover codecs from the input bytes and output extension. Instantiate only the selected codec factories.
- Measure end-to-end throughput and the expression evaluator independently before changing either for performance.
- Express supported features and resource limits in the API. Reject representations that would require implicit precision loss or dropping transparency.
