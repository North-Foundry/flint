#!/usr/bin/env python3
"""Deterministic PNG->JPEG and in-process Flint benchmark harness (stdlib only)."""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path


PATTERNS = ("gradient", "checker", "noise")


def chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def make_png(path: Path, width: int, height: int, pattern: str) -> None:
    """Write RGB8 PNG using only stdlib; each pixel is a stable function of x/y."""
    rows = bytearray()
    for y in range(height):
        rows.append(0)  # PNG filter: None
        for x in range(width):
            if pattern == "gradient":
                rgb = (
                    x * 255 // max(1, width - 1),
                    y * 255 // max(1, height - 1),
                    (x + y) * 255 // max(1, width + height - 2),
                )
            elif pattern == "checker":
                v = 235 if ((x // 16) ^ (y // 16)) & 1 else 20
                rgb = (v, (v + x % 31) & 255, (v + y % 47) & 255)
            else:
                # Integer hash; independent of Python's randomized hash seed.
                n = (x * 0x9E3779B1 + y * 0x85EBCA77 + 0xC2B2AE3D) & 0xFFFFFFFF
                n ^= n >> 16
                n = (n * 0x7FEB352D) & 0xFFFFFFFF
                n ^= n >> 15
                rgb = (n & 255, (n >> 8) & 255, (n >> 16) & 255)
            rows.extend(rgb)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows, 6))
        + chunk(b"IEND", b"")
    )


def worker(argv: list[str]) -> int:
    """Run one process in isolation so RUSAGE_CHILDREN reports its peak RSS."""
    try:
        import resource
    except ImportError:
        resource = None
    started = time.perf_counter()
    proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    elapsed = time.perf_counter() - started
    peak = None
    if resource:
        peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        # macOS reports bytes. Linux commonly reports KiB.
        if sys.platform != "darwin":
            peak *= 1024
    print(json.dumps({
        "returncode": proc.returncode,
        "seconds": elapsed,
        "peak_rss_bytes": peak,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }))
    return 0


def run_one(argv: list[str]) -> dict:
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", *argv],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {"returncode": result.returncode, "stderr": result.stderr}
    return json.loads(result.stdout)


def key_values(output: str) -> dict[str, str]:
    return dict(re.findall(r"^([a-z_]+)=(.*)$", output, re.MULTILINE))


def comparison_rows(rows: list[dict]) -> list[dict]:
    """Pair successful optimization modes and calculate checksum-guarded speedups."""
    grouped: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        if row.get("kind") != "in_process_ops":
            continue
        key = (row["operation"], row["width"], row["height"], row["requested_backend"], row["repetition"])
        grouped.setdefault(key, {})[row["optimization"]] = row
    comparisons = []
    for key, modes in grouped.items():
        disabled, automatic = modes.get("Disabled"), modes.get("Auto")
        if not disabled or not automatic:
            continue
        checksum_match = (
            disabled.get("ok") and automatic.get("ok")
            and disabled["checksum"] == automatic["checksum"]
            and disabled.get("selected_backend") == automatic.get("selected_backend")
        )
        comparisons.append({
            "operation": key[0], "width": key[1], "height": key[2],
            "requested_backend": key[3], "repetition": key[4],
            "disabled_selected_backend": disabled.get("selected_backend"),
            "auto_selected_backend": automatic.get("selected_backend"),
            "checksum_match": bool(checksum_match),
            "speedup": disabled["seconds"] / automatic["seconds"]
                if checksum_match and automatic["seconds"] else None,
            "disabled_seconds": disabled.get("seconds"), "auto_seconds": automatic.get("seconds"),
            "disabled_planning_seconds": disabled.get("planning_seconds"),
            "auto_planning_seconds": automatic.get("planning_seconds"),
            "disabled_peak_managed_bytes": disabled.get("peak_managed_bytes"),
            "auto_peak_managed_bytes": automatic.get("peak_managed_bytes"),
            "disabled_estimated_managed_bytes": disabled.get("estimated_managed_bytes"),
            "auto_estimated_managed_bytes": automatic.get("estimated_managed_bytes"),
            "disabled_source_instruction_count": disabled.get("source_instruction_count"),
            "auto_source_instruction_count": automatic.get("source_instruction_count"),
            "disabled_instruction_count": disabled.get("instruction_count"),
            "auto_instruction_count": automatic.get("instruction_count"),
            "disabled_folded_instructions": disabled.get("folded_instructions"),
            "auto_folded_instructions": automatic.get("folded_instructions"),
            "disabled_evaluator_workspace_bytes": disabled.get("evaluator_workspace_bytes"),
            "auto_evaluator_workspace_bytes": automatic.get("evaluator_workspace_bytes"),
            "disabled_peak_rss_bytes": disabled.get("peak_rss_bytes"),
            "auto_peak_rss_bytes": automatic.get("peak_rss_bytes"),
        })
    return comparisons


def jpeg_size(path: Path) -> int | None:
    try:
        size = path.stat().st_size
        if size < 4:
            return None
        with path.open("rb") as jpeg:
            if jpeg.read(2) != b"\xff\xd8":
                return None
            jpeg.seek(-2, os.SEEK_END)
            return size if jpeg.read(2) == b"\xff\xd9" else None
    except OSError:
        return None


def tool_commands() -> list[tuple[str, list[str]]]:
    found = []
    vips = shutil.which("vips")
    if vips:
        found.append(("libvips", [vips, "jpegsave", "{input}", "{output}", "--Q", "85", "--subsample-mode", "on"]))
    else:
        print("SKIP libvips: vips executable not found", file=sys.stderr)
    magick = shutil.which("magick") or shutil.which("convert")
    if magick:
        found.append(("ImageMagick", [magick, "{input}", "-sampling-factor", "4:2:0", "-quality", "85", "{output}"]))
    else:
        print("SKIP ImageMagick: magick/convert executable not found", file=sys.stderr)
    php = shutil.which("php")
    if php:
        check = subprocess.run([php, "-r", "exit(extension_loaded('gd') ? 0 : 1);"], capture_output=True)
        if check.returncode == 0:
            # GD uses its library default chroma sampling; it has no portable API to set it.
            found.append(("PHP GD", [php, "{php_script}", "{input}", "{output}"]))
        else:
            print("SKIP PHP GD: PHP is installed without the GD extension", file=sys.stderr)
    else:
        print("SKIP PHP GD: php executable not found", file=sys.stderr)
    return found


def version(command: str, args: list[str]) -> str | None:
    executable = shutil.which(command)
    if not executable:
        return None
    try:
        result = subprocess.run([executable, *args], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return None
        return (result.stdout or result.stderr).splitlines()[0].strip()
    except (OSError, subprocess.TimeoutExpired, IndexError):
        return None


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        return worker(sys.argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/output"))
    parser.add_argument("--flint-convert", nargs="+", help="conversion command; use {input} and {output} placeholders")
    parser.add_argument("--flint-ops", nargs="+", help="in-process example command; use {width}, {height}, {backend}, {operation}, {iterations}")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=3, help="fresh-process repetitions per in-process case")
    parser.add_argument("--width", type=int, default=256, help="fixed width for memory scaling")
    parser.add_argument("--full-matrix", action="store_true", help="include long chains and edge image dimensions")
    parser.add_argument("--include-8k", action="store_true", help="add a minimal 8192x8192 in-process sample")
    parser.add_argument("--baseline-matrix", action="store_true", help="use original dimensions and workload set")
    parser.add_argument("--folding-matrix", action="store_true", help="run only constant-folding in-process workloads and the codec matrix")
    args = parser.parse_args()
    if args.iterations < 1 or args.repetitions < 1 or args.width < 1:
        parser.error("iterations, repetitions, and width must be positive")
    if (args.full_matrix or args.include_8k) and args.baseline_matrix:
        parser.error("expanded matrices and --baseline-matrix are mutually exclusive")
    if args.folding_matrix and (args.full_matrix or args.include_8k or args.baseline_matrix):
        parser.error("--folding-matrix cannot be combined with other workload flags")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands = tool_commands()
    flint_manifest = Path("crates/flint/Cargo.toml")
    convert_bin = ops_bin = None
    if flint_manifest.exists() and (not args.flint_convert or not args.flint_ops):
        build = subprocess.run(
            ["cargo", "build", "--locked", "--release", "--manifest-path", str(flint_manifest),
             "--example", "convert", "--example", "bench_ops"],
            capture_output=True,
            text=True,
        )
        binary_dir = Path(os.environ.get("CARGO_TARGET_DIR", "target")) / "release" / "examples"
        suffix = ".exe" if os.name == "nt" else ""
        convert_bin, ops_bin = binary_dir / f"convert{suffix}", binary_dir / f"bench_ops{suffix}"
        if build.returncode or not convert_bin.exists() or not ops_bin.exists():
            print(f"SKIP Flint examples: cargo build failed or binaries are missing\n{build.stderr.strip()}", file=sys.stderr)
            convert_bin = ops_bin = None
    if args.flint_convert:
        commands.insert(0, ("Flint", args.flint_convert))
    elif convert_bin:
        commands.insert(0, ("Flint", [str(convert_bin), "{input}", "{output}"]))
    else:
        print("SKIP Flint conversion: pass --flint-convert or provide crates/flint/Cargo.toml", file=sys.stderr)
    supports_folding = False
    flint_ops = args.flint_ops
    if not flint_ops and ops_bin:
        flint_ops = [str(ops_bin), "{width}", "{height}", "{backend}", "{operation}", "{iterations}", "{optimization}"]
        probe = run_one([str(ops_bin), "1", "1", "Scalar", "fold_simple", "1", "Disabled"])
        supports_folding = probe.get("returncode") == 0 and "folded_instructions" in key_values(probe.get("stdout", ""))
    if args.folding_matrix and not args.flint_ops and not supports_folding:
        parser.error("--folding-matrix requires a bench_ops build with folding workloads or a compatible --flint-ops command")
    if not flint_ops:
        print("SKIP Flint in-process operations: pass --flint-ops or add the Cargo package", file=sys.stderr)
    versions = {
        "libvips": version("vips", ["--version"]),
        "ImageMagick": version("magick", ["-version"]) or version("convert", ["-version"]),
        "PHP": version("php", ["--version"]),
        "PHP GD": version("php", ["-r", "echo gd_info()['GD Version'];"]),
        "Python": sys.version.split()[0],
        "Rust": version("rustc", ["--version"]),
        "Cargo": version("cargo", ["--version"]),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }

    rows = []
    dimensions = [(640, 480), (1920, 1080), *((args.width, h) for h in (512, 2048, 8192))]
    if args.full_matrix:
        dimensions = [(640, 480), (1920, 1080), (3840, 2160),
                      (16, 16), (1, 257), (17, 9), (4096, 1),
                      (args.width, 512), (args.width, 2048), (args.width, 8192)]
    if args.include_8k:
        dimensions.append((8192, 8192))
    # Codec dimensions stay fixed regardless of the evaluator matrix selection.
    fixtures = [(w, h, p) for w, h in ((640, 480), (1920, 1080)) for p in PATTERNS]
    fixtures.extend((args.width, h, "noise") for h in (512, 2048, 8192))
    with tempfile.TemporaryDirectory(prefix="flint-bench-") as temp:
        tempdir = Path(temp)
        php_script = tempdir / "gd_convert.php"
        php_script.write_text(
            "<?php\n"
            "$im = imagecreatefrompng($argv[1]);\n"
            "if (!$im) { fwrite(STDERR, 'PNG decode failed'); exit(2); }\n"
            "if (!imagejpeg($im, $argv[2], 85)) { fwrite(STDERR, 'JPEG encode failed'); exit(2); }\n"
        )
        for width, height, pattern in fixtures:
            source = tempdir / f"{width}x{height}-{pattern}.png"
            make_png(source, width, height, pattern)
            for name, template in commands:
                for repetition in range(args.iterations):
                    output = tempdir / f"{name.replace(' ', '-')}-{width}x{height}-{pattern}-{repetition}.jpg"
                    argv = [
                        s.format(input=source, output=output, php_script=php_script)
                        for s in template
                    ]
                    report = run_one(argv)
                    size = jpeg_size(output)
                    ok = report.get("returncode") == 0 and size is not None
                    error = report.get("stderr") if report.get("returncode") else None
                    if report.get("returncode") == 0 and not ok:
                        error = "converter exited successfully but produced no valid JPEG (SOI/EOI markers missing)"
                    seconds = report.get("seconds")
                    details = key_values(report.get("stdout", "")) if ok else {}
                    rows.append({
                        "kind": "png_to_jpeg",
                        "tool": name,
                        "width": width,
                        "height": height,
                        "pattern": pattern,
                        "iteration": repetition,
                        "ok": ok,
                        "seconds": seconds,
                        "megapixels_per_second": width * height / 1e6 / seconds if seconds else None,
                        "peak_rss_bytes": report.get("peak_rss_bytes"),
                        "peak_managed_bytes": int(details["peak_managed_bytes"])
                            if "peak_managed_bytes" in details else None,
                        "estimated_managed_bytes": int(details["estimated_managed_bytes"])
                            if "estimated_managed_bytes" in details else None,
                        "selected_backend": details.get("backend"),
                        "output_bytes": size if ok else None,
                        "returncode": report.get("returncode"),
                        "error": error,
                    })
                    if not ok:
                        print(
                            f"FAILED run: {name} {width}x{height} {pattern}: {error or 'missing output'}",
                            file=sys.stderr,
                        )

        if flint_ops:
            legacy_operations = ("grayscale", "blur", "combined")
            folding_operations = ("fold_simple", "fold_nested", "fold_repeated", "fold_chain",
                                  "fold_multichannel", "fold_stencil", "fold_none")
            standard_operations = ("brightness", "grayscale", "blur", "combined", "chain2", "color_mix", "mixed",
                                   "brightness_grayscale_brightness", "color_grayscale_contrast", "chain_blur_chain")
            if supports_folding:
                standard_operations += folding_operations
            chain_operations = tuple(name for length in (5, 10, 20, 50)
                                     for name in (f"chain{length}", f"expensive_chain{length}")) + ("expensive_chain2",)
            supports_mode = any("{optimization}" in part for part in flint_ops)
            if args.folding_matrix:
                matrix_cases = [(dimension, folding_operations) for dimension in dimensions[:2]]
            elif args.baseline_matrix:
                matrix_cases = [(dimension, legacy_operations) for dimension in dimensions]
            elif args.full_matrix:
                matrix_cases = [
                    ((640, 480), standard_operations + chain_operations + ("expensive",)),
                    ((1920, 1080), ("brightness", "grayscale", "chain2", "chain5", "chain10",
                                    "color_grayscale_contrast", "chain_blur_chain")),
                    ((3840, 2160), ("brightness", "grayscale", "chain5", "color_mix",
                                    "color_grayscale_contrast", "chain_blur_chain")),
                    *((dimension, ("brightness", "grayscale", "chain2"))
                      for dimension in ((16, 16), (1, 257), (17, 9), (4096, 1),
                                        (args.width, 512), (args.width, 2048), (args.width, 8192))),
                ]
                if args.include_8k:
                    matrix_cases.append(((8192, 8192), ("grayscale",)))
            elif args.include_8k:
                matrix_cases = [((8192, 8192), ("grayscale",))]
            elif not supports_mode:
                # Preserve the historical workload contract for existing adapters.
                matrix_cases = [(dimension, legacy_operations) for dimension in dimensions]
            else:
                matrix_cases = [(dimension, standard_operations) for dimension in dimensions[:2]]
            mode_sweep = ("Disabled", "Auto") if supports_mode else ("Auto",)
            for (width, height), operations in matrix_cases:
                for backend in ("Scalar", "Simd"):
                    for operation in operations:
                        for optimization in mode_sweep:
                            for repetition in range(args.repetitions):
                                argv = [
                                    part.format(width=width, height=height, backend=backend,
                                                operation=operation, iterations=args.iterations,
                                                optimization=optimization)
                                    for part in flint_ops
                                ]
                                report = run_one(argv)
                                details = key_values(report.get("stdout", ""))
                                required = {"elapsed_seconds", "megapixels_per_second", "backend",
                                            "peak_managed_bytes", "estimated_managed_bytes", "checksum"}
                                if supports_folding:
                                    required.update({"source_instruction_count", "instruction_count",
                                                     "folded_instructions", "execution_folded_instructions",
                                                     "evaluator_workspace_bytes"})
                                error = report.get("stderr") if report.get("returncode") else None
                                if report.get("returncode") == 0 and not required <= details.keys():
                                    error = "missing benchmark output fields"
                                if supports_folding and report.get("returncode") == 0 and error is None:
                                    if details["folded_instructions"] != details["execution_folded_instructions"]:
                                        error = "planned and executed fold counts differ"
                                ok = report.get("returncode") == 0 and error is None
                                def numeric(name: str) -> float | None:
                                    return float(details[name]) if ok and name in details else None
                                def integer(name: str) -> int | None:
                                    return int(details[name]) if ok and name in details else None
                                rows.append({
                                    "kind": "in_process_ops", "tool": "Flint",
                                    "requested_backend": backend, "selected_backend": details.get("backend"),
                                    "optimization": details.get("optimization", optimization),
                                    "operation": operation, "width": width, "height": height,
                                    "repetition": repetition, "ok": ok,
                                    "seconds": numeric("elapsed_seconds"),
                                    "megapixels_per_second": numeric("megapixels_per_second"),
                                    "planning_seconds": numeric("planning_seconds"),
                                    "stages": integer("stages"), "kernels": integer("kernels"),
                                    "instruction_count": integer("instruction_count"),
                                    "source_instruction_count": integer("source_instruction_count"),
                                    "folded_instructions": integer("folded_instructions"),
                                    "execution_folded_instructions": integer("execution_folded_instructions"),
                                    "temporary_workspace_bytes": integer("temporary_workspace_bytes"),
                                    "evaluator_workspace_bytes": integer("evaluator_workspace_bytes"),
                                    "peak_rss_bytes": report.get("peak_rss_bytes"),
                                    "peak_managed_bytes": integer("peak_managed_bytes"),
                                    "estimated_managed_bytes": integer("estimated_managed_bytes"),
                                    "checksum": details.get("checksum"),
                                    "returncode": report.get("returncode"), "error": error,
                                })
                                if not ok:
                                    print(f"FAILED run: {backend} {optimization} {operation} {width}x{height}: {error}", file=sys.stderr)
    result_path = args.output_dir / "results.json"
    result_path.write_text(json.dumps({
        "settings": {
            "jpeg_quality": 85,
            "sampling_factor": "4:2:0 for Flint/libvips/ImageMagick, PHP GD default",
            "iterations": args.iterations,
            "repetitions": args.repetitions,
            "full_matrix": args.full_matrix,
            "include_8k": args.include_8k,
            "baseline_matrix": args.baseline_matrix,
            "folding_matrix": args.folding_matrix,
            "fixtures": fixtures,
            "internal_dimensions": dimensions,
            "internal_pixels": "synthetic deterministic RGB8 rows, generated incrementally in Rust",
        },
        "versions": versions,
        "results": rows,
        "comparisons": comparison_rows(rows),
    }, indent=2) + "\n")
    print(result_path)
    has_flint = bool((args.flint_convert or convert_bin) and flint_ops)
    checksum_failure = any(not comparison["checksum_match"] for comparison in comparison_rows(rows))
    return int(not has_flint or any(not row["ok"] for row in rows) or checksum_failure)


if __name__ == "__main__":
    raise SystemExit(main())
