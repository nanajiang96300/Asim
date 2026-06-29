#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.evaluate_ldl_quality import bits_to_16qam, demod_16qam, estimate_se, ls_channel_estimate
from scripts.DeepUnfold.evaluate_bj_iterative_vs_chol_ldl import (
    average_symbol_energy,
    build_block_richardson_preconditioner,
    chebyshev_omega,
    chebyshev_omega_adaptive,
    generate_channel,
    make_square_qam_constellation,
)


@dataclass
class NumericExecConfig:
    nr: int
    nt: int
    n_sc: int
    batch: int
    trials: int
    snr_db_list: List[float]
    pilot_len: int
    pilot_snr_db: float | None
    seed: int
    out_dir: Path
    bj_layers: int
    bj_block: int
    bj_adaptive_bounds: bool
    bj_iter_weight: bool
    dump_max_samples: int


def parse_model_json(path: Path) -> Dict[str, int | bool]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    model = obj["models"][0]
    attrs = model.get("attributes", {})

    def as_bool(val: str, default: bool = False) -> bool:
        if val is None:
            return default
        return str(val).strip().lower() in {"1", "true", "on", "yes"}

    return {
        "nr": int(model.get("matrix_m", 64)),
        "nt": int(model.get("matrix_k", 16)),
        "layers": int(attrs.get("layers", "8")),
        "block": int(attrs.get("block_size", "2")),
        "adaptive": as_bool(attrs.get("adaptive_bounds", "1"), True),
        "iter_weight": as_bool(attrs.get("iter_weight", "1"), True),
    }


def bj_inverse_with_trace(a_mat: np.ndarray, layers: int, blk: int, adaptive: bool, iter_weight: bool) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    b_mat, b_inv = build_block_richardson_preconditioner(a_mat, blk=blk, precond_solver="cholesky")
    y_mat = np.zeros_like(a_mat, dtype=np.complex128)
    identity = np.eye(a_mat.shape[0], dtype=np.complex128)

    if iter_weight:
        omegas = chebyshev_omega_adaptive(b_mat, n_layers=layers, nt=a_mat.shape[0]) if adaptive else chebyshev_omega(n_layers=layers, bmin=0.1, bmax=1.2)
    else:
        omegas = [1.0] * max(layers, 1)

    y_layers = []
    residual_norms = []
    for omega in omegas:
        residual = identity - b_mat @ y_mat
        residual_norms.append(float(np.linalg.norm(residual)))
        y_mat = y_mat + omega * residual
        y_layers.append(np.asarray(y_mat, dtype=np.complex128))

    a_inv = y_mat @ b_inv
    payload = {
        "a_est": np.asarray(a_mat, dtype=np.complex128),
        "b_mat": np.asarray(b_mat, dtype=np.complex128),
        "b_inv": np.asarray(b_inv, dtype=np.complex128),
        "omegas": np.asarray(omegas, dtype=np.float64),
        "residual_norms": np.asarray(residual_norms, dtype=np.float64),
        "y_last": np.asarray(y_mat, dtype=np.complex128),
        "y_layers": np.stack(y_layers, axis=0) if y_layers else np.zeros((0, a_mat.shape[0], a_mat.shape[1]), dtype=np.complex128),
    }
    return a_inv, payload


def run_numeric_exec(cfg: NumericExecConfig) -> tuple[Path, Path]:
    rng = np.random.default_rng(cfg.seed)
    constellation = make_square_qam_constellation(16)
    symbol_energy = average_symbol_energy(constellation)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    dump_dir = cfg.out_dir / "internal_dump"
    dump_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    manifest_rows = []
    dump_count = 0

    for snr_db in cfg.snr_db_list:
        snr_lin = 10.0 ** (snr_db / 10.0)
        noise_var = 1.0 / snr_lin
        pilot_snr_lin = snr_lin if cfg.pilot_snr_db is None else 10.0 ** (cfg.pilot_snr_db / 10.0)
        pilot_noise_var = 1.0 / pilot_snr_lin

        se_sum = 0.0
        total_samples = cfg.trials * cfg.batch * cfg.n_sc
        for sample_idx in range(total_samples):
            h_true = generate_channel(rng, cfg.nr, cfg.nt)
            h_est = ls_channel_estimate(rng, h_true, cfg.pilot_len, pilot_noise_var)

            tx_bits = rng.integers(0, 2, size=(cfg.nt, 4), dtype=np.int32)
            x_tx = bits_to_16qam(tx_bits)
            noise = np.sqrt(noise_var / 2.0) * (rng.standard_normal(cfg.nr) + 1j * rng.standard_normal(cfg.nr))
            y_rx = h_true @ x_tx + noise

            a_est = h_est.conj().T @ h_est + (noise_var / symbol_energy) * np.eye(cfg.nt, dtype=np.complex128)
            h_h_est = h_est.conj().T

            a_inv, payload = bj_inverse_with_trace(
                a_est,
                layers=cfg.bj_layers,
                blk=cfg.bj_block,
                adaptive=cfg.bj_adaptive_bounds,
                iter_weight=cfg.bj_iter_weight,
            )
            w = a_inv @ h_h_est
            x_hat = w @ y_rx
            _ = demod_16qam(x_hat)
            se_sum += estimate_se(w, h_true, noise_var)

            if dump_count < cfg.dump_max_samples:
                f = dump_dir / f"bj_numeric_snr{snr_db:g}_sample{sample_idx:04d}.npz"
                np.savez_compressed(f, **payload)
                manifest_rows.append(
                    {
                        "snr_db": float(snr_db),
                        "sample_idx": sample_idx,
                        "file": str(f),
                        "layers": cfg.bj_layers,
                        "block_size": cfg.bj_block,
                        "adaptive_bounds": int(cfg.bj_adaptive_bounds),
                        "iter_weight": int(cfg.bj_iter_weight),
                    }
                )
                dump_count += 1

        rows.append({"snr_db": float(snr_db), "se_block_jacobi_numeric": se_sum / max(total_samples, 1)})
        print(f"[done] numeric-exec snr={snr_db} dB -> se_block_jacobi_numeric={rows[-1]['se_block_jacobi_numeric']:.4f}")

    csv_path = cfg.out_dir / "se_block_jacobi_numeric_exec.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["snr_db", "se_block_jacobi_numeric"])
        writer.writeheader()
        writer.writerows(rows)

    manifest_path = dump_dir / "manifest.csv"
    if manifest_rows:
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)
    else:
        manifest_path.write_text("snr_db,sample_idx,file,layers,block_size,adaptive_bounds,iter_weight\n", encoding="utf-8")

    return csv_path, manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Block Jacobi numeric execution mode (Python reference)")
    parser.add_argument("--model-json", type=str, default="example/block_jacobi_test.json")
    parser.add_argument("--nr", type=int, default=None)
    parser.add_argument("--nt", type=int, default=None)
    parser.add_argument("--n-sc", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--snr-db", type=str, default="0,5,10")
    parser.add_argument("--pilot-len", type=int, default=None)
    parser.add_argument("--pilot-snr-db", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="result_new/bj_numeric_exec")
    parser.add_argument("--dump-max-samples", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_json)
    if not model_path.is_absolute():
        model_path = Path(PROJECT_ROOT) / model_path

    meta = parse_model_json(model_path)
    nr = int(args.nr) if args.nr is not None else int(meta["nr"])
    nt = int(args.nt) if args.nt is not None else int(meta["nt"])
    pilot_len = int(args.pilot_len) if args.pilot_len is not None else nt

    cfg = NumericExecConfig(
        nr=nr,
        nt=nt,
        n_sc=args.n_sc,
        batch=args.batch,
        trials=args.trials,
        snr_db_list=[float(x.strip()) for x in args.snr_db.split(",") if x.strip()],
        pilot_len=pilot_len,
        pilot_snr_db=args.pilot_snr_db,
        seed=args.seed,
        out_dir=Path(args.out_dir),
        bj_layers=int(meta["layers"]),
        bj_block=int(meta["block"]),
        bj_adaptive_bounds=bool(meta["adaptive"]),
        bj_iter_weight=bool(meta["iter_weight"]),
        dump_max_samples=args.dump_max_samples,
    )

    csv_path, manifest_path = run_numeric_exec(cfg)
    print("done")
    print(f"se_csv={csv_path}")
    print(f"dump_manifest={manifest_path}")


if __name__ == "__main__":
    main()
