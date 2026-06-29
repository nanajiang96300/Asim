#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "result_new"
REF = ROOT / "result_new_true"
SIM = ROOT / "build_asim" / "bin" / "Simulator"
CFG = ROOT / "configs" / "ascend_910b_quiet.json"


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def regenerate_traces() -> None:
    cases = [
        ("cholesky_test", ROOT / "example" / "cholesky_test.json", RESULT / "cholesky" / "block"),
        ("cholesky_noblock_test", ROOT / "example" / "cholesky_noblock_test.json", RESULT / "cholesky" / "noblock"),
        ("ldl_test", ROOT / "example" / "ldl_test_moderate3.json", RESULT / "ldl" / "block"),
        ("ldl_noblock_test", ROOT / "example" / "ldl_noblock_test.json", RESULT / "ldl" / "noblock"),
    ]
    for mode, model_json, out_dir in cases:
        out_dir.mkdir(parents=True, exist_ok=True)
        env = {
            **os.environ,
            "ONNXIM_TRACE_CSV": str(out_dir / "trace.csv"),
            "ONNXIM_MAX_CORE_CYCLES": "20000000",
        }
        with (out_dir / "run.log").open("w") as f:
            subprocess.run(
                [
                    str(SIM),
                    "--config",
                    str(CFG),
                    "--models_list",
                    str(model_json),
                    "--mode",
                    mode,
                ],
                check=True,
                cwd=ROOT,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
            )


def regenerate_core0_timelines() -> None:
    traces = [
        RESULT / "cholesky" / "block" / "trace.csv",
        RESULT / "cholesky" / "noblock" / "trace.csv",
        RESULT / "ldl" / "block" / "trace.csv",
        RESULT / "ldl" / "noblock" / "trace.csv",
        RESULT / "block_richardson" / "operator" / "trace.csv",
    ]
    for trace in traces:
        out_png = trace.parent / "timeline_core0_unified.png"
        run(
            [
                "python3",
                "visualizer_png.py",
                "-i",
                str(trace),
                "-o",
                str(out_png),
                "--split-cube-wait-track",
                "--core-filter",
                "Core0",
            ]
        )


def export_four_group_tables() -> None:
    jobs = [
        ("scripts/export_cholesky_cycle_table.py", RESULT / "cholesky" / "block" / "trace.csv", RESULT / "cholesky" / "block" / "detailed_cycles_v3.csv"),
        ("scripts/export_cholesky_cycle_table.py", RESULT / "cholesky" / "noblock" / "trace.csv", RESULT / "cholesky" / "noblock" / "detailed_cycles_v3.csv"),
        ("scripts/export_ldl_cycle_table.py", RESULT / "ldl" / "block" / "trace.csv", RESULT / "ldl" / "block" / "detailed_cycles_v3.csv"),
        ("scripts/export_ldl_cycle_table.py", RESULT / "ldl" / "noblock" / "trace.csv", RESULT / "ldl" / "noblock" / "detailed_cycles_v3.csv"),
    ]
    for script, trace, out_csv in jobs:
        run(
            [
                "python3",
                script,
                "--trace",
                str(trace),
                "--output",
                str(out_csv),
                "--matrix-m",
                "64",
                "--matrix-u",
                "16",
                "--reducer",
                "median",
            ]
        )


def copy_reference_block_jacobi_tables() -> None:
    src_dir = REF / "block_richardson" / "operator"
    dst_dir = RESULT / "block_richardson" / "operator"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "block_jacobi_cycle_detail.csv",
        "block_jacobi_cycle_detail_major_summary.csv",
        "block_jacobi_cycle_major_summary.csv",
    ]:
        (dst_dir / name).write_text((src_dir / name).read_text())


def event_group(key: str) -> str:
    return re.sub(r"\d+", "*", key)


def formula_template(formula: str) -> str:
    return re.sub(r"\d+", "n", formula)


def compact_one(input_csv: Path, output_csv: Path) -> tuple[int, int]:
    rows = list(csv.DictReader(input_csv.open(newline="")))
    total_cycles_all = sum(float(r["compute_cycles"]) for r in rows)

    buckets: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        g = event_group(row["event_key"])
        k = (row["operator_mode"], g, row["compute_op"])
        buckets.setdefault(k, []).append(row)

    out_rows: list[dict[str, str]] = []
    for (operator_mode, group, comp), items in sorted(
        buckets.items(), key=lambda kv: sum(float(x["compute_cycles"]) for x in kv[1]), reverse=True
    ):
        cycles = [float(x["compute_cycles"]) for x in items]
        total_cycles = sum(cycles)
        keys = sorted({x["event_key"] for x in items})
        out_rows.append(
            {
                "operator_mode": operator_mode,
                "event_group": group,
                "compute_op": comp,
                "row_count": str(len(items)),
                "event_key_count": str(len(keys)),
                "total_cycles": f"{total_cycles:.2f}",
                "mean_cycles": f"{statistics.mean(cycles):.2f}",
                "median_cycles": f"{statistics.median(cycles):.2f}",
                "min_cycles": f"{min(cycles):.2f}",
                "max_cycles": f"{max(cycles):.2f}",
                "share_pct": f"{(100.0 * total_cycles / total_cycles_all) if total_cycles_all > 0 else 0.0:.2f}",
                "overall_total_cycles": f"{total_cycles_all:.2f}",
                "example_event_keys": "|".join(keys[:5]),
                "formula_template": formula_template(items[0]["formula"]),
            }
        )

    out_rows.append(
        {
            "operator_mode": rows[0]["operator_mode"] if rows else "",
            "event_group": "__TOTAL__",
            "compute_op": "ALL",
            "row_count": str(len(rows)),
            "event_key_count": str(len({r['event_key'] for r in rows})),
            "total_cycles": f"{total_cycles_all:.2f}",
            "mean_cycles": "",
            "median_cycles": "",
            "min_cycles": "",
            "max_cycles": "",
            "share_pct": "100.00",
            "overall_total_cycles": f"{total_cycles_all:.2f}",
            "example_event_keys": "",
            "formula_template": "",
        }
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "operator_mode",
                "event_group",
                "compute_op",
                "row_count",
                "event_key_count",
                "total_cycles",
                "mean_cycles",
                "median_cycles",
                "min_cycles",
                "max_cycles",
                "share_pct",
                "overall_total_cycles",
                "example_event_keys",
                "formula_template",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    return len(rows), len(out_rows)


def build_compact_and_index() -> None:
    mappings = [
        (RESULT / "cholesky" / "block" / "detailed_cycles_v3.csv", RESULT / "cholesky" / "block" / "detailed_cycles_v3_compact_stats.csv"),
        (RESULT / "cholesky" / "noblock" / "detailed_cycles_v3.csv", RESULT / "cholesky" / "noblock" / "detailed_cycles_v3_compact_stats.csv"),
        (RESULT / "ldl" / "block" / "detailed_cycles_v3.csv", RESULT / "ldl" / "block" / "detailed_cycles_v3_compact_stats.csv"),
        (RESULT / "ldl" / "noblock" / "detailed_cycles_v3.csv", RESULT / "ldl" / "noblock" / "detailed_cycles_v3_compact_stats.csv"),
        (RESULT / "block_richardson" / "operator" / "block_jacobi_cycle_detail.csv", RESULT / "block_richardson" / "operator" / "block_jacobi_cycle_detail_compact_stats.csv"),
    ]

    idx_rows = []
    for src, dst in mappings:
        src_rows, out_rows = compact_one(src, dst)
        idx_rows.append(
            {
                "input_csv": str(src),
                "output_csv": str(dst),
                "source_rows": str(src_rows),
                "compact_rows": str(out_rows),
            }
        )

    with (RESULT / "operator_compact_summary_index.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["input_csv", "output_csv", "source_rows", "compact_rows"])
        writer.writeheader()
        writer.writerows(idx_rows)


def summarize_trace(trace_csv: Path) -> tuple[int, int, dict[str, float]]:
    rows = list(csv.DictReader(trace_csv.open(newline="")))
    global_max_end = max(int(r["end_cycle"]) for r in rows) if rows else 0
    core0 = [r for r in rows if r["unit"].startswith("Core0_")]
    core0_max_end = max(int(r["end_cycle"]) for r in core0) if core0 else 0

    totals = {
        "Scalar": 0,
        "Vector": 0,
        "Cube": 0,
        "MTE2": 0,
        "Wait": 0,
        "MTE3": 0,
    }
    sum_all = 0
    for r in core0:
        dur = int(r["end_cycle"]) - int(r["start_cycle"])
        if dur <= 0:
            continue
        sum_all += dur
        unit = r["unit"].split("_", 1)[1]
        if unit in totals:
            totals[unit] += dur

    pct = {k: (100.0 * v / sum_all if sum_all > 0 else 0.0) for k, v in totals.items()}
    return global_max_end, core0_max_end, pct


def build_4group_summary() -> None:
    rows = []
    cfg = [
        ("cholesky_block", RESULT / "cholesky" / "block" / "trace.csv"),
        ("cholesky_noblock", RESULT / "cholesky" / "noblock" / "trace.csv"),
        ("ldl_block", RESULT / "ldl" / "block" / "trace.csv"),
        ("ldl_noblock", RESULT / "ldl" / "noblock" / "trace.csv"),
    ]
    for group, trace in cfg:
        gmax, cmax, pct = summarize_trace(trace)
        rows.append(
            {
                "group": group,
                "global_max_end": str(gmax),
                "core0_max_end": str(cmax),
                "core0_scalar_share_pct": f"{pct['Scalar']:.2f}",
                "core0_vector_share_pct": f"{pct['Vector']:.2f}",
                "core0_cube_share_pct": f"{pct['Cube']:.2f}",
                "core0_mte2_share_pct": f"{pct['MTE2']:.2f}",
                "core0_wait_share_pct": f"{pct['Wait']:.2f}",
                "core0_mte3_share_pct": f"{pct['MTE3']:.2f}",
            }
        )

    with (RESULT / "summary_4groups_latest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "global_max_end",
                "core0_max_end",
                "core0_scalar_share_pct",
                "core0_vector_share_pct",
                "core0_cube_share_pct",
                "core0_mte2_share_pct",
                "core0_wait_share_pct",
                "core0_mte3_share_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    regenerate_traces()
    regenerate_core0_timelines()
    export_four_group_tables()
    copy_reference_block_jacobi_tables()
    build_compact_and_index()
    build_4group_summary()
    print("done: rebuilt result_new formula stats with result_new_true-compatible structure")


if __name__ == "__main__":
    main()
