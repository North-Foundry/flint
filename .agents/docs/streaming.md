# Streaming

For output row `y`, a kernel requests its upstream rows through `clamp(y + max_y)`. Its input ring retains the span from `min_y` to `max_y`, capped at the image height. Each ring stores planar `f32` rows, including RGB. Horizontal coordinates clamp to the nearest edge in the evaluator. A pointwise kernel needs one input row for the whole group of original stages.

The decoder fills a runtime-owned packed byte row. With operations, the runtime unpacks it into the source ring and packs the final output row for the encoder. A direct transcode reuses the packed row across both codecs. The runtime drains any unread source rows before finishing the decoder, including when a stencil only samples earlier rows.

The memory limit covers reusable packed rows, float rings, and scalar and SIMD evaluator scratch. Standalone kernels use scratch as an operand stack and fused kernels use it for temporary values. Graph metadata, planning allocations, executable program storage, allocator overhead, and codec-owned state are outside the limit. Width and neighborhood size bound working pixel storage even as height grows. Allocation size arithmetic is checked before use.

`Engine::open_source` takes a factory for repeatable streams at byte zero. Open may read ahead while inspecting metadata. The factory must provide the same image on every execution. Changed dimensions or pixel format give an error when the source is reopened. File output is staged next to its destination and published only after both codecs finish. An arbitrary `Write` may contain partial data after a failure.
