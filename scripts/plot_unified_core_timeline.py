#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run_plot(input_csv: Path, output_png: Path, core: str, visualizer: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3",
        str(visualizer),
        "-i",
        str(input_csv),
        "-o",
        str(output_png),
        "--core-filter",
        core,
        "--force-cube-gap-wait",
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render unified one-core timelines for BlockJacobi and Cholesky with identical style."
    )
    parser.add_argument("--core", default="Core0", help="Core name to render, default Core0")
    parser.add_argument(
        "--block-jacobi-trace",
        default="/project/Asim/result_new/block_jacobi/operator/trace.csv",
        help="BlockJacobi trace CSV path",
    )
    parser.add_argument(
        "--cholesky-trace",
        default="/project/Asim/result_new/cholesky/operator/trace.csv",
        help="Cholesky trace CSV path",
    )
    parser.add_argument(
        "--block-jacobi-out",
        default="/project/Asim/result_new/block_jacobi/operator/timeline_core0_unified.png",
        help="BlockJacobi output PNG path",
    )
    parser.add_argument(
        "--cholesky-out",
        default="/project/Asim/result_new/cholesky/operator/timeline_core0_unified.png",
        help="Cholesky output PNG path",
    )
    parser.add_argument(
        "--visualizer",
        default="/project/Asim/visualizer_png.py",
        help="Path to visualizer_png.py",
    )
    args = parser.parse_args()

    block_jacobi_trace = Path(args.block_jacobi_trace)
    cholesky_trace = Path(args.cholesky_trace)
    block_jacobi_out = Path(args.block_jacobi_out)
    cholesky_out = Path(args.cholesky_out)
    visualizer = Path(args.visualizer)

    if not block_jacobi_trace.exists():
        raise SystemExit(f"Missing BlockJacobi trace: {block_jacobi_trace}")
    if not cholesky_trace.exists():
        raise SystemExit(f"Missing Cholesky trace: {cholesky_trace}")
    if not visualizer.exists():
        raise SystemExit(f"Missing visualizer script: {visualizer}")

    run_plot(block_jacobi_trace, block_jacobi_out, args.core, visualizer)
    run_plot(cholesky_trace, cholesky_out, args.core, visualizer)

    print("Unified timelines generated:")
    print(f"- {block_jacobi_out}")
    print(f"- {cholesky_out}")


if __name__ == "__main__":
    main()
