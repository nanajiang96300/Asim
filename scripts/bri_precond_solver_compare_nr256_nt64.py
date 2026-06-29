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
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "build_asim" / "bin" / "Simulator"
CFG = ROOT / "configs" / "ascend_910b_quiet.json"
RESULT_ROOT = ROOT / "result_new"

if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from skill.scripts.skill_experiment_pipeline import export_block_jacobi_cycle_detail
from scripts.DeepUnfold.evaluate_bj_iterative_vs_chol_ldl import CompareConfig as BjAlgoConfig, run_compare


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run BJ preconditioner solver comparison for nr256_nt64")
    p.add_argument("--tag", type=str, default="bj_precond_compare_nr256_nt64")
    p.add_argument("--nr", type=int, default=256)
    p.add_argument("--nt", type=int, default=64)
    p.add_argument("--blocks", type=str, default="2,4,8,16,32,64")
    p.add_argument("--snr-db", type=str, default="0,5,10,15,20,25,30")
    p.add_argument("--modulation", type=str, default="64qam", choices=["16qam", "64qam", "bpsk"])

    p.add_argument("--sim-batch-size", type=int, default=96)
    p.add_argument("--max-core-cycles", type=int, default=20000000)

    p.add_argument("--se-n-sc", type=int, default=8)
    p.add_argument("--se-batch", type=int, default=2)
    p.add_argument("--se-trials", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trunc-mantissa-bits", type=int, default=10)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not SIM.exists():
        raise FileNotFoundError(f"Simulator not found at: {SIM}")
    if not CFG.exists():
        raise FileNotFoundError(f"Config not found at: {CFG}")

    blocks = [int(x.strip()) for x in args.blocks.split(",") if x.strip()]
    snr_list = [float(x.strip()) for x in args.snr_db.split(",") if x.strip()]

    nr = int(args.nr)
    nt = int(args.nt)

    exp_dir = make_exp_dir(args.tag)
    generated_cfg = exp_dir / "generated_configs"

    summary_rows: List[dict] = []

    for blk in blocks:
        if blk > nt:
            continue

        solvers = ["direct2x2"] if blk == 2 else ["direct", "cholesky"]
        for precond_solver in solvers:
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

    summary_csv = exp_dir / "bj_precond_compare_summary.csv"
    fieldnames = ["nr", "nt", "block_size", "precond_solver", "finish_cycle"] + [f"se_{int(s)}" for s in snr_list]
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"experiment_dir={exp_dir}")
    print(f"summary_csv={summary_csv}")


if __name__ == "__main__":
    main()
