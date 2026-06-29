#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def render_one(visualizer: Path, trace_csv: Path, output_png: Path, core: str) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3",
        str(visualizer),
        "-i",
        str(trace_csv),
        "-o",
        str(output_png),
        "--core-filter",
        core,
        "--force-cube-gap-wait",
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-generate unified one-core timeline plots for all traces under result_new."
    )
    parser.add_argument("--result-root", default="/project/Asim/result_new", help="Root folder to scan")
    parser.add_argument("--visualizer", default="/project/Asim/visualizer_png.py", help="Path to visualizer_png.py")
    parser.add_argument("--core", default="Core0", help="Core to plot, default Core0")
    parser.add_argument(
        "--output-name",
        default="timeline_core0_unified.png",
        help="Output PNG file name in each trace directory",
    )
    args = parser.parse_args()

    root = Path(args.result_root)
    visualizer = Path(args.visualizer)
    if not root.exists():
        raise SystemExit(f"result root not found: {root}")
    if not visualizer.exists():
        raise SystemExit(f"visualizer not found: {visualizer}")

    traces = sorted(root.glob("**/trace.csv"))
    if not traces:
        raise SystemExit(f"no trace.csv found under {root}")

    generated = []
    for trace in traces:
        out = trace.parent / args.output_name
        render_one(visualizer, trace, out, args.core)
        generated.append(out)

    print(f"generated_count={len(generated)}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
