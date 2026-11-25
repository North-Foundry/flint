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
    parser.add_argument("--width", type=int, default=256, help="fixed width for memory scaling")
    args = parser.parse_args()
    if args.iterations < 1 or args.width < 1:
        parser.error("iterations and width must be positive")
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
    flint_ops = args.flint_ops
    if not flint_ops and ops_bin:
        flint_ops = [str(ops_bin), "{width}", "{height}", "{backend}", "{operation}", "{iterations}"]
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
    fixtures = [(w, h, p) for w, h in dimensions[:2] for p in PATTERNS]
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
            for width, height in dimensions:
                for backend in ("Scalar", "Simd"):
                    for operation in ("grayscale", "blur", "combined"):
                        argv = [
                            s.format(width=width, height=height, backend=backend,
                                     operation=operation, iterations=args.iterations)
                            for s in flint_ops
                        ]
                        report = run_one(argv)
                        details = key_values(report.get("stdout", ""))
                        required = {"elapsed_seconds", "megapixels_per_second", "backend",
                                    "peak_managed_bytes", "estimated_managed_bytes", "checksum"}
                        error = report.get("stderr") if report.get("returncode") else None
                        if report.get("returncode") == 0 and not required <= details.keys():
                            error = "missing benchmark output fields"
                        ok = report.get("returncode") == 0 and error is None
                        rows.append({
                            "kind": "in_process_ops",
                            "tool": "Flint",
                            "requested_backend": backend,
                            "selected_backend": details.get("backend"),
                            "operation": operation,
                            "width": width,
                            "height": height,
                            "ok": ok,
                            "seconds": float(details["elapsed_seconds"]) if ok else None,
                            "megapixels_per_second": float(details["megapixels_per_second"]) if ok else None,
                            "peak_rss_bytes": report.get("peak_rss_bytes"),
                            "peak_managed_bytes": int(details["peak_managed_bytes"]) if ok else None,
                            "estimated_managed_bytes": int(details["estimated_managed_bytes"]) if ok else None,
                            "checksum": details.get("checksum"),
                            "returncode": report.get("returncode"),
                            "error": error,
                        })
                        if not ok:
                            print(f"FAILED run: {backend} {operation} {width}x{height}: {error}", file=sys.stderr)
    result_path = args.output_dir / "results.json"
    result_path.write_text(json.dumps({
        "settings": {
            "jpeg_quality": 85,
            "sampling_factor": "4:2:0 for Flint/libvips/ImageMagick, PHP GD default",
            "iterations": args.iterations,
            "fixtures": fixtures,
            "internal_dimensions": dimensions,
            "internal_pixels": "synthetic deterministic RGB8 rows, generated incrementally in Rust",
        },
        "versions": versions,
        "results": rows,
    }, indent=2) + "\n")
    print(result_path)
    has_flint = bool((args.flint_convert or convert_bin) and flint_ops)
    return int(not has_flint or any(not row["ok"] for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
