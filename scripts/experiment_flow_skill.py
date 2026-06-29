#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class AlgorithmSpec:
    name: str
    family: str
    variant: str
    trace_path: str
    matrix_m: int = 64
    matrix_u: int = 16
    reducer: str = "median"
    core_prefix: str = ""
    simulate_cmd: str = ""
    formula_role: str = ""


def run_cmd(args: List[str], cwd: Path = PROJECT_ROOT) -> None:
    subprocess.run(args, cwd=str(cwd), check=True)


def run_shell(command: str, cwd: Path = PROJECT_ROOT) -> None:
    subprocess.run(command, cwd=str(cwd), shell=True, check=True)


def load_config(path: Path) -> List[AlgorithmSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    specs = []
    for item in payload.get("algorithms", []):
        specs.append(
            AlgorithmSpec(
                name=item["name"],
                family=item["family"],
                variant=item["variant"],
                trace_path=item["trace_path"],
                matrix_m=int(item.get("matrix_m", 64)),
                matrix_u=int(item.get("matrix_u", 16)),
                reducer=str(item.get("reducer", "median")),
                core_prefix=str(item.get("core_prefix", "")),
                simulate_cmd=str(item.get("simulate_cmd", "")),
                formula_role=str(item.get("formula_role", "")),
            )
        )
    if not specs:
        raise ValueError("Config has no algorithms.")
    return specs


def infer_formula_role(spec: AlgorithmSpec) -> str:
    if spec.formula_role:
        return spec.formula_role
    key = f"{spec.family.lower()}_{spec.variant.lower()}"
    mapping = {
        "cholesky_block": "cholesky_block",
        "cholesky_noblock": "cholesky_noblock",
        "ldl_block": "ldl_block",
        "ldl_noblock": "ldl_noblock",
        "block_jacobi_block": "jacobi_block",
        "jacobi_block": "jacobi_block",
    }
    return mapping.get(key, "")


def export_cycle_tables(spec: AlgorithmSpec, trace_csv: Path, out_dir: Path) -> Path:
    family = spec.family.lower()
    if family == "cholesky":
        detail_csv = out_dir / "detailed_cycles_v3.csv"
        run_cmd(
            [
                "python3",
                "scripts/export_cholesky_cycle_table.py",
                "--trace",
                str(trace_csv),
                "--output",
                str(detail_csv),
                "--mode",
                "chol_block" if spec.variant.lower() == "block" else "chol_nb",
                "--matrix-m",
                str(spec.matrix_m),
                "--matrix-u",
                str(spec.matrix_u),
                "--reducer",
                spec.reducer,
                "--core-prefix",
                spec.core_prefix,
            ]
        )
        return detail_csv

    if family == "ldl":
        detail_csv = out_dir / "detailed_cycles_v3.csv"
        run_cmd(
            [
                "python3",
                "scripts/export_ldl_cycle_table.py",
                "--trace",
                str(trace_csv),
                "--output",
                str(detail_csv),
                "--mode",
                "ldl_block" if spec.variant.lower() == "block" else "ldl_noblock",
                "--matrix-m",
                str(spec.matrix_m),
                "--matrix-u",
                str(spec.matrix_u),
                "--reducer",
                spec.reducer,
                "--core-prefix",
                spec.core_prefix,
            ]
        )
        return detail_csv

    if family in {"block_jacobi", "jacobi"}:
        detail_csv = out_dir / "block_jacobi_cycle_detail.csv"
        run_cmd(
            [
                "python3",
                "scripts/export_operator_cycle_table.py",
                "--trace",
                str(trace_csv),
                "--output",
                str(detail_csv),
                "--mode",
                "block_jacobi",
                "--matrix-m",
                str(spec.matrix_m),
                "--matrix-u",
                str(spec.matrix_u),
                "--reducer",
                spec.reducer,
                "--core-prefix",
                spec.core_prefix,
            ]
        )
        return detail_csv

    raise ValueError(f"Unsupported family: {spec.family}")


def render_timeline(trace_csv: Path, out_png: Path, core: str) -> None:
    run_cmd(
        [
            "python3",
            "visualizer_png.py",
            "-i",
            str(trace_csv),
            "-o",
            str(out_png),
            "--core-filter",
            core,
            "--split-cube-wait-track",
        ]
    )


def export_step_stats(trace_csv: Path, out_csv: Path, core_prefix: str) -> None:
    run_cmd(
        [
            "python3",
            "scripts/export_trace_step_stats.py",
            "--trace",
            str(trace_csv),
            "--output",
            str(out_csv),
            "--core-prefix",
            core_prefix,
        ]
    )


def export_compact(algo_dir: Path) -> None:
    run_cmd(["python3", "scripts/export_compact_operator_stats.py", "--root", str(algo_dir)])


def parse_total_cycles(compact_csv: Path) -> float:
    with compact_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    total = [row for row in rows if row.get("event_group") == "__TOTAL__"]
    if not total:
        return 0.0
    return float(total[0].get("total_cycles", "0") or 0)


def run_se_eval(out_root: Path, formula_map: Dict[str, Path], snr_db: str) -> Optional[Path]:
    required = ["cholesky_noblock", "cholesky_block", "ldl_noblock", "ldl_block"]
    if not all(key in formula_map for key in required):
        return None

    se_out = out_root / "se"
    args = [
        "python3",
        "scripts/validate_se_block_noblock_rayleigh.py",
        "--snr-db",
        snr_db,
        "--formula-csv-cholesky-noblock",
        str(formula_map["cholesky_noblock"]),
        "--formula-csv-cholesky-block",
        str(formula_map["cholesky_block"]),
        "--formula-csv-ldl-noblock",
        str(formula_map["ldl_noblock"]),
        "--formula-csv-ldl-block",
        str(formula_map["ldl_block"]),
        "--out-dir",
        str(se_out),
    ]
    if "jacobi_block" in formula_map:
        args.extend(["--formula-csv-jacobi-block", str(formula_map["jacobi_block"])])
    run_cmd(args)
    return se_out / "se_three_algorithms_block_noblock_rayleigh.csv"


def write_report(
    report_path: Path,
    run_id: str,
    artifacts: List[Dict[str, str]],
    se_csv: Optional[Path],
) -> None:
    lines: List[str] = []
    lines.append(f"# 实验流程报告：{run_id}")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("- 产物根目录：`result_new`（本次在 `result_new/{run_id}`）")
    lines.append("")
    lines.append("## 1) 算子产物清单")
    lines.append("")
    lines.append("| 算法 | trace | 详细周期表 | 压缩周期表 | 总周期 | 时序图 |")
    lines.append("|---|---|---|---|---:|---|")
    for item in artifacts:
        lines.append(
            f"| {item['name']} | `{item['trace']}` | `{item['detail_csv']}` | `{item['compact_csv']}` | {item['total_cycles']} | `{item['timeline_png']}` |"
        )

    lines.append("")
    lines.append("## 2) 公式周期统计说明")
    lines.append("")
    lines.append("- 详细表：逐公式步骤周期 (`detailed_cycles_v3.csv` / `block_jacobi_cycle_detail.csv`)。")
    lines.append("- 压缩表：按事件模板归并（例如 `L_UPDATE_*_*_PACK*`）并包含 `__TOTAL__` 对账行。")

    if se_csv and se_csv.exists():
        with se_csv.open("r", newline="", encoding="utf-8") as f:
            se_rows = list(csv.DictReader(f))
        lines.append("")
        lines.append("## 3) SE 结果与合理性检查")
        lines.append("")
        lines.append(f"- SE CSV: `{se_csv}`")
        if se_rows:
            cols = [c for c in se_rows[0].keys() if c.startswith("se_")]
            lines.append("")
            lines.append("| 指标 | 单调性(SNR递增) |")
            lines.append("|---|---|")
            for col in cols:
                values = [float(r[col]) for r in se_rows]
                mono = all(values[i] <= values[i + 1] + 1e-9 for i in range(len(values) - 1))
                lines.append(f"| {col} | {'PASS' if mono else 'WARN'} |")

    lines.append("")
    lines.append("## 4) 本次流程覆盖")
    lines.append("")
    lines.append("1. 跑出周期 CSV（详细 + major summary + step stats）")
    lines.append("2. 绘制时序图（Core0）")
    lines.append("3. 公式周期统计与压缩汇总")
    lines.append("4. 基于公式统计执行 SE 验证（若四个基础公式齐全）")
    lines.append("5. 自动生成本报告")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment flow skill: one-command reproducible pipeline.")
    parser.add_argument("--config", required=True, help="JSON config path")
    parser.add_argument("--run-id", required=True, help="Run id under result_new, e.g. 20260409_run01")
    parser.add_argument("--result-root", default="result_new")
    parser.add_argument("--timeline-core", default="Core0")
    parser.add_argument("--snr-db", default="0,5,10,15,20")
    parser.add_argument("--skip-sim", action="store_true", help="Skip simulate_cmd even if provided")
    args = parser.parse_args()

    specs = load_config(Path(args.config))

    result_root = Path(args.result_root)
    if not result_root.is_absolute():
        result_root = (PROJECT_ROOT / result_root).resolve()
    run_root = result_root / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)

    copied_config = run_root / "skill_config_snapshot.json"
    copied_config.write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")

    formula_map: Dict[str, Path] = {}
    artifacts: List[Dict[str, str]] = []

    for spec in specs:
        algo_dir = run_root / spec.name
        algo_dir.mkdir(parents=True, exist_ok=True)

        if spec.simulate_cmd and (not args.skip_sim):
            run_shell(spec.simulate_cmd)

        src_trace = Path(spec.trace_path)
        if not src_trace.is_absolute():
            src_trace = (PROJECT_ROOT / src_trace).resolve()
        if not src_trace.exists():
            raise FileNotFoundError(f"Trace not found: {src_trace}")

        trace_csv = algo_dir / "trace.csv"
        shutil.copyfile(src_trace, trace_csv)

        detail_csv = export_cycle_tables(spec, trace_csv, algo_dir)
        timeline_png = algo_dir / "timeline_core0_latest.png"
        render_timeline(trace_csv, timeline_png, args.timeline_core)

        step_stats_csv = algo_dir / "step_stats.csv"
        export_step_stats(trace_csv, step_stats_csv, spec.core_prefix)

        export_compact(algo_dir)
        compact_csv = detail_csv.with_name(f"{detail_csv.stem}_compact_stats.csv")

        role = infer_formula_role(spec)
        if role:
            formula_map[role] = detail_csv

        artifacts.append(
            {
                "name": spec.name,
                "trace": str(trace_csv.relative_to(PROJECT_ROOT)),
                "detail_csv": str(detail_csv.relative_to(PROJECT_ROOT)),
                "compact_csv": str(compact_csv.relative_to(PROJECT_ROOT)),
                "timeline_png": str(timeline_png.relative_to(PROJECT_ROOT)),
                "total_cycles": f"{parse_total_cycles(compact_csv):.2f}",
            }
        )

    se_csv = run_se_eval(run_root, formula_map, args.snr_db)
    report_path = run_root / "EXPERIMENT_REPORT.md"
    write_report(report_path, args.run_id, artifacts, se_csv)

    print(f"[done] run_root={run_root}")
    print(f"[done] report={report_path}")
    if se_csv:
        print(f"[done] se_csv={se_csv}")
    else:
        print("[warn] SE step skipped: missing required formula roles.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
