# Symbolic primitives

| Primitive | Closure input | Output |
| --- | --- | --- |
| `map_color` | The active channel as an `Expr` | The same pixel format, with each channel mapped independently |
| `map_pixel` | `PixelSamples` with symbolic `r`, `g`, and `b` fields | `Pixel::gray` or `Pixel::rgb` |
| `stencil` | `Neighborhood` with `at(dx, dy)` | The same pixel format, computed independently per channel |

Closures run once during graph construction. A Gray8 input supplies the same sample for all three `map_pixel` fields. A stencil reads from the previous stage and uses clamp-to-edge on both axes. Offsets are integers bounded by `MAX_STENCIL_OFFSET`.

`flint-basic` shows all three contracts. Brightness maps and clamps each channel. Grayscale uses weighted encoded RGB samples. Gaussian blur generates a normalized kernel and composes horizontal and vertical stencil stages.
