# Streaming

For output row `y`, a stage requests its upstream rows through `clamp(y + max_y)`. Its input ring retains the span from `min_y` to `max_y`, capped at the image height. Each ring stores planar `f32` rows, including RGB. Horizontal coordinates clamp to the nearest edge in the evaluator. Pointwise stages need one input row each.

The decoder fills a runtime-owned packed byte row. With operations, the runtime unpacks it into the source ring and packs the final output row for the encoder. A direct transcode reuses the packed row across both codecs. The runtime drains any unread source rows before finishing the decoder, including when a stencil only samples earlier rows.

The memory limit covers the reusable packed rows, float rings, and evaluator stack. It excludes graph metadata, plan bookkeeping, allocator overhead, and codec-owned state. Width and neighborhood size bound working pixel storage even as height grows. Allocation size arithmetic is checked before use.

`Engine::open_source` takes a factory for repeatable streams at byte zero. Open may read ahead while inspecting metadata. The factory must provide the same image on every execution. Changed dimensions or pixel format give an error when the source is reopened. File output is staged next to its destination and published only after both codecs finish. An arbitrary `Write` may contain partial data after a failure.
