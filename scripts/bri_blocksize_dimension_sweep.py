#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "build_asim" / "bin" / "Simulator"
CFG = ROOT / "configs" / "ascend_910b_quiet.json"
RESULT_ROOT = ROOT / "result_new"

if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from skill.scripts.skill_experiment_pipeline import export_block_jacobi_cycle_detail
from scripts.DeepUnfold.evaluate_bj_iterative_vs_chol_ldl import CompareConfig as BjAlgoConfig, run_compare


def run_cmd(cmd: List[str], cwd: Path = ROOT, env: dict | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def make_exp_dir(tag: str) -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULT_ROOT / f"exp_{ts}_{tag}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_model_json(path: Path, name: str, m: int, k: int, batch_size: int, attrs: Dict[str, str]) -> None:
    obj = {
        "models": [
            {
                "name": name,
                "batch_size": batch_size,
                "matrix_m": m,
                "matrix_k": k,
                "attributes": attrs,
            }
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=4), encoding="utf-8")


def parse_finish_cycle(run_log: Path) -> int:
    text = run_log.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"finish at (\d+)", text)
    if not match:
        raise RuntimeError(f"finish cycle not found in {run_log}")
    return int(match.group(1))


def run_sim_block_jacobi(case_dir: Path, model_json: Path, max_core_cycles: int, block_size: int, precond_solver: str) -> int:
    op_dir = case_dir / "block_richardson" / "operator"
    op_dir.mkdir(parents=True, exist_ok=True)
    trace = op_dir / "trace.csv"
    run_log = op_dir / "run.log"

    env = {
        **os.environ,
        "ONNXIM_TRACE_CSV": str(trace),
        "ONNXIM_MAX_CORE_CYCLES": str(max_core_cycles),
    }

    with run_log.open("w", encoding="utf-8") as logf:
        subprocess.run(
            [
                str(SIM),
                "--config",
                str(CFG),
                "--models_list",
                str(model_json),
                "--mode",
                "block_jacobi_test",
            ],
            cwd=ROOT,
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=True,
        )

    export_block_jacobi_cycle_detail(
        case_dir,
        nt=int(json.loads(model_json.read_text(encoding="utf-8"))["models"][0]["matrix_k"]),
        block_size=block_size,
        precond_solver=precond_solver,
    )
    return parse_finish_cycle(run_log)


def run_bj_se(
    case_dir: Path,
    nr: int,
    nt: int,
    blk: int,
    precond_solver: str,
    snr_db_list: List[float],
    n_sc: int,
    batch: int,
    trials: int,
    seed: int,
    trunc_mantissa_bits: int,
    modulation: str,
) -> Dict[float, float]:
    out_dir = case_dir / "se_compare"
    cfg = BjAlgoConfig(
        nr=nr,
        nt=nt,
        n_sc=n_sc,
        batch=batch,
        trials=trials,
        block_size=2,
        snr_db_list=snr_db_list,
        pilot_len=nt,
        pilot_snr_db=None,
        modulation=modulation,
        seed=seed,
        out_dir=str(out_dir),
        num_format="fp16",
        reciprocal_mode="approx",
        trunc_mantissa_bits=trunc_mantissa_bits,
        mac_chunk=4,
        bj_layers=8,
        bj_block=blk,
        bj_adaptive_bounds=True,
        bj_precond_solver=precond_solver,
        bj_omega_policy="classic",
        bj_omega_tail_scale=1.0,
        bj_corr_steps=0,
        progress_every=0,
        watchdog_timeout_sec=0.0,
        max_seconds_per_snr=0.0,
    )
    rows = run_compare(cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "se_block_jacobi_only.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["snr_db", "se_block_jacobi"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"snr_db": row["snr_db"], "se_block_jacobi": row["se_bj_iterative"]})

    se_map: Dict[float, float] = {}
    for row in rows:
        se_map[float(row["snr_db"])] = float(row["se_bj_iterative"])
    return se_map


def draw_plots(exp_dir: Path, summary_rows: List[dict], snr_list: List[float]) -> Tuple[Path, Path, Path]:
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    def plot_se(filtered_rows, out_png, title):
        plt.figure(figsize=(10.5, 5.6))
        for r in filtered_rows:
            label = f"{r['nr']}x{r['nt']} B{r['block_size']}"
            ys = [float(r.get(f"se_{int(s)}", "nan")) for s in snr_list]
            plt.plot(snr_list, ys, marker="o", label=label)
        plt.xlabel("SNR (dB)")
        plt.ylabel("SE (bits/s/Hz)")
        plt.title(title)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

    se_png_direct = plots_dir / "bj_direct_se.png"
    se_png_chol = plots_dir / "bj_cholesky_se.png"

    direct_rows = [r for r in summary_rows if r['precond_solver'] in ('direct', 'direct2x2')]
    chol_rows = [r for r in summary_rows if r['precond_solver'] in ('cholesky', 'direct2x2')]

    plot_se(direct_rows, se_png_direct, "Block-Jacobi SE vs SNR (Direct Inversion)")
    plot_se(chol_rows, se_png_chol, "Block-Jacobi SE vs SNR (Cholesky Inversion)")

    cycle_png = plots_dir / "bj_blocksize_finish_cycle.png"
    plt.figure(figsize=(14.0, 6.2))
    cycles = [int(r["finish_cycle"]) for r in summary_rows]
    labels = [f"{r['nr']}x{r['nt']}\nB{r['block_size']} ({r['precond_solver']})" for r in summary_rows]
    plt.bar(range(len(labels)), cycles)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Finish Cycle")
    plt.title("Block-Jacobi Finish Cycle")
    plt.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(cycle_png, dpi=200)
    plt.close()

    return cycle_png, se_png_direct, se_png_chol


def write_report(
    exp_dir: Path,
    summary_rows: List[dict],
    snr_list: List[float],
    cycle_png: Path,
    se_png_direct: Path,
    se_png_chol: Path,
) -> Path:
    report = exp_dir / "BJ_BLOCKSIZE_DIMENSION_REPORT.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# BJ 块大小与维度联合测试报告\n\n")
        f.write("- 规则: 直接求逆图使用 `direct/direct2x2`；Cholesky 图使用 `cholesky`（并保留 `2x2` 参考点）。\n")
        f.write("- 测试对象: Block-Jacobi 算子（周期）+ 重建数值 BJ-SE\n\n")

        f.write("## 汇总表\n\n")
        header = ["nr", "nt", "block_size", "precond_solver", "finish_cycle"] + [f"se_{int(s)}" for s in snr_list]
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for row in summary_rows:
            vals = [str(row[h]) for h in header]
            f.write("| " + " | ".join(vals) + " |\n")

        f.write("\n## 图像\n\n")
        f.write(f"![finish_cycle]({cycle_png.relative_to(exp_dir).as_posix()})\n\n")
        f.write(f"![se_direct]({se_png_direct.relative_to(exp_dir).as_posix()})\n\n")
        f.write(f"![se_cholesky]({se_png_chol.relative_to(exp_dir).as_posix()})\n\n")

        f.write("## 关键结论\n\n")
        for (nr, nt) in sorted({(int(r['nr']), int(r['nt'])) for r in summary_rows}):
            sub = [r for r in summary_rows if int(r["nr"]) == nr and int(r["nt"]) == nt]
            best_cycle = min(sub, key=lambda x: int(x["finish_cycle"]))
            best_se20 = max(sub, key=lambda x: float(x.get("se_20", "-1e9")))
            f.write(
                f"- `{nr}x{nt}`: 最低周期为 `B{best_cycle['block_size']}` (`{best_cycle['finish_cycle']}`)，"
                f"20dB 最高 SE 为 `B{best_se20['block_size']}` (`{float(best_se20.get('se_20', 'nan')):.6f}`)。\n"
            )

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BJ block-size sweep across selected dimensions")
    parser.add_argument("--tag", type=str, default="bj_blocksize_sweep")
    parser.add_argument("--snr-db", type=str, default="0,5,10,15,20,25,30")
    parser.add_argument("--se-n-sc", type=int, default=8)
    parser.add_argument("--se-batch", type=int, default=2)
    parser.add_argument("--se-trials", type=int, default=1)
    parser.add_argument("--modulation", type=str, default="64qam", choices=["16qam", "64qam", "bpsk"])
    parser.add_argument("--sim-batch-size", type=int, default=96)
    parser.add_argument("--max-core-cycles", type=int, default=20000000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trunc-mantissa-bits", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snr_list = [float(x.strip()) for x in args.snr_db.split(",") if x.strip()]

    combos: List[Tuple[int, int, int, str]] = []
    base_combos = [
        (64, 16, 2),
        (64, 16, 4),
        (64, 16, 8),
        (64, 16, 16),
        (128, 32, 2),
        (128, 32, 4),
        (128, 32, 8),
        (128, 32, 16),
        (128, 32, 32),
    ]
    for nr, nt, blk in base_combos:
        if blk == 2:
            combos.append((nr, nt, blk, "direct2x2"))
        else:
            combos.append((nr, nt, blk, "direct"))
            combos.append((nr, nt, blk, "cholesky"))

    exp_dir = make_exp_dir(args.tag)
    generated_cfg = exp_dir / "generated_configs"

    summary_rows: List[dict] = []

    for nr, nt, blk, precond_solver in combos:
        case_key = f"nr{nr}_nt{nt}_b{blk}_{precond_solver}"
        case_dir = exp_dir / case_key
        model_json = generated_cfg / f"{case_key}.json"

        
        attrs = {
            "layers": "8",
            "block_size": str(blk),
            "group_sync": "16",
            "adaptive_bounds": "1",
            "iter_weight": "1",
            "omega_relaxed": "1",
            "fused_by_gemm": "1",
            "by_preload_period": "16",
            "fuse_residual_update": "1",
            "by_kernel_fuse_factor": "4",
            "precond_solver": precond_solver,
        }

        write_model_json(model_json, f"block_jacobi_{nt}x{nt}_b{blk}", nt, nt, args.sim_batch_size, attrs)

        finish_cycle = run_sim_block_jacobi(
            case_dir,
            model_json,
            args.max_core_cycles,
            block_size=blk,
            precond_solver=precond_solver,
        )
        se_map = run_bj_se(
            case_dir=case_dir,
            nr=nr,
            nt=nt,
            blk=blk,
            precond_solver=precond_solver,
            snr_db_list=snr_list,
            n_sc=args.se_n_sc,
            batch=args.se_batch,
            trials=args.se_trials,
            seed=args.seed,
            trunc_mantissa_bits=args.trunc_mantissa_bits,
            modulation=args.modulation,
        )

        row = {
            "nr": nr,
            "nt": nt,
            "block_size": blk,
            "precond_solver": precond_solver,
            "finish_cycle": finish_cycle,
        }
        for snr in snr_list:
            row[f"se_{int(snr)}"] = f"{se_map.get(snr, float('nan')):.6f}"
        summary_rows.append(row)

    summary_csv = exp_dir / "bj_blocksize_dimension_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["nr", "nt", "block_size", "precond_solver", "finish_cycle"] + [f"se_{int(s)}" for s in snr_list]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    cycle_png, se_png_direct, se_png_chol = draw_plots(exp_dir, summary_rows, snr_list)
    report = write_report(exp_dir, summary_rows, snr_list, cycle_png, se_png_direct, se_png_chol)

    print(f"experiment_dir={exp_dir}")
    print(f"summary_csv={summary_csv}")
    print(f"cycle_png={cycle_png}")
    print(f"se_png_direct={se_png_direct}")
    print(f"se_png_cholesky={se_png_chol}")
    print(f"report={report}")


if __name__ == "__main__":
    main()
