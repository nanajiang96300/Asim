#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.DeepUnfold.bj_deep_unfolding_onnxim import (
    bj_chebyshev_inverse,
    build_regularized_system,
    make_square_qam_constellation,
)
from scripts.evaluate_ldl_quality import (
    bits_to_16qam,
    demod_16qam,
    estimate_se,
    ls_channel_estimate,
)


Array = np.ndarray


@dataclass
class SimConfig:
    nr: int
    nt: int
    n_sc: int
    batch: int
    trials: int
    snr_db_list: List[float]
    pilot_len: int
    pilot_snr_db: float | None
    modulation: str
    seed: int
    bj_layers: int
    bj_block: int
    bj_adaptive_bounds: bool
    out_dir: str


def generate_channel(rng: np.random.Generator, nr: int, nt: int) -> Array:
    return (rng.standard_normal((nr, nt)) + 1j * rng.standard_normal((nr, nt))) / np.sqrt(2.0)


def average_symbol_energy(constellation: Array) -> float:
    return float(np.mean(np.abs(constellation) ** 2))


def run_simulation(cfg: SimConfig) -> List[Dict[str, float]]:
    rng = np.random.default_rng(cfg.seed)
    constellation = make_square_qam_constellation(16) if cfg.modulation == "16qam" else None
    symbol_energy = average_symbol_energy(constellation) if constellation is not None else 1.0

    metrics: List[Dict[str, float]] = []
    t_global = time.time()

    for idx, snr_db in enumerate(cfg.snr_db_list, start=1):
        t_snr = time.time()
        print(f"[progress] SNR {snr_db} dB ({idx}/{len(cfg.snr_db_list)}) started...", flush=True)

        snr_lin = 10.0 ** (snr_db / 10.0)
        noise_var = 1.0 / snr_lin
        pilot_snr_lin = snr_lin if cfg.pilot_snr_db is None else 10.0 ** (cfg.pilot_snr_db / 10.0)
        pilot_noise_var = 1.0 / pilot_snr_lin

        ber_err = 0
        total_bits = 0
        se_sum = 0.0

        total_samples = cfg.trials * cfg.batch * cfg.n_sc

        for _ in range(total_samples):
            h_true = generate_channel(rng, cfg.nr, cfg.nt)
            h_est = ls_channel_estimate(rng, h_true, cfg.pilot_len, pilot_noise_var)

            if cfg.modulation == "16qam":
                tx_bits = rng.integers(0, 2, size=(cfg.nt, 4), dtype=np.int32)
                x_tx = bits_to_16qam(tx_bits)
                bits_per_sym = 4
            else:
                tx_bits = rng.integers(0, 2, size=(cfg.nt, 1), dtype=np.int32)
                x_tx = (2 * tx_bits[:, 0] - 1).astype(np.float64).astype(np.complex128)
                bits_per_sym = 1

            noise = np.sqrt(noise_var / 2.0) * (
                rng.standard_normal(cfg.nr) + 1j * rng.standard_normal(cfg.nr)
            )
            y_rx = h_true @ x_tx + noise

            _, a_est = build_regularized_system(h_est, noise_var, constellation)
            a_inv_bj = bj_chebyshev_inverse(
                a_est,
                n_layers=cfg.bj_layers,
                adaptive_bounds=cfg.bj_adaptive_bounds,
            )

            w_bj = a_inv_bj @ h_est.conj().T
            xhat_bj = w_bj @ y_rx

            if cfg.modulation == "16qam":
                bits_hat_bj = demod_16qam(xhat_bj)
            else:
                bits_hat_bj = (np.real(xhat_bj) >= 0).astype(np.int32)[:, None]

            ber_err += int(np.sum(bits_hat_bj != tx_bits))
            total_bits += cfg.nt * bits_per_sym
            se_sum += estimate_se(w_bj, h_true, noise_var)

        ber = ber_err / max(total_bits, 1)
        se = se_sum / max(total_samples, 1)
        metrics.append(
            {
                "snr_db": snr_db,
                "ber_block_jacobi": ber,
                "se_block_jacobi": se,
            }
        )

        print(
            f"[progress] SNR {snr_db} dB done in {time.time() - t_snr:.2f}s, elapsed {time.time() - t_global:.2f}s",
            flush=True,
        )

    return metrics


def save_outputs(cfg: SimConfig, metrics: List[Dict[str, float]]) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)

    csv_path = os.path.join(cfg.out_dir, "block_jacobi_ber_se_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["snr_db", "ber_block_jacobi", "se_block_jacobi"])
        writer.writeheader()
        writer.writerows(metrics)

    snr = [item["snr_db"] for item in metrics]
    ber = [max(item["ber_block_jacobi"], 1e-8) for item in metrics]
    se = [item["se_block_jacobi"] for item in metrics]

    ber_png = os.path.join(cfg.out_dir, "block_jacobi_ber_vs_snr.png")
    plt.figure(figsize=(7.2, 5.0))
    plt.semilogy(snr, ber, marker="o", linestyle="-", label="Block Jacobi")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("Block Jacobi BER vs SNR")
    plt.grid(True, which="both", linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ber_png, dpi=220)
    plt.close()

    se_png = os.path.join(cfg.out_dir, "block_jacobi_se_vs_snr.png")
    plt.figure(figsize=(7.2, 5.0))
    plt.plot(snr, se, marker="o", linestyle="-", label="Block Jacobi")
    plt.xlabel("SNR (dB)")
    plt.ylabel("SE (bits/s/Hz)")
    plt.title("Block Jacobi SE vs SNR")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(se_png, dpi=220)
    plt.close()

    report_path = os.path.join(cfg.out_dir, "block_jacobi_ber_se_report.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# Block Jacobi BER/SE 仿真报告\n\n")
        handle.write("## 算法来源\n")
        handle.write("- 复用 `scripts/DeepUnfold/bj_deep_unfolding_onnxim.py` 中 Block-Jacobi + Chebyshev 反演实现。\n")
        handle.write("- BER/SE 评估口径复用 `scripts/evaluate_ldl_quality.py`。\n\n")
        handle.write("## 参数\n")
        handle.write(f"- nr={cfg.nr}, nt={cfg.nt}, n_sc={cfg.n_sc}, batch={cfg.batch}, trials={cfg.trials}\n")
        handle.write(f"- snr_db={cfg.snr_db_list}\n")
        handle.write(f"- modulation={cfg.modulation}, pilot_len={cfg.pilot_len}, pilot_snr_db={cfg.pilot_snr_db}\n")
        handle.write(f"- bj_layers={cfg.bj_layers}, bj_block={cfg.bj_block}, adaptive_bounds={cfg.bj_adaptive_bounds}\n\n")
        handle.write("## 输出文件\n")
        handle.write(f"- `block_jacobi_ber_se_metrics.csv`\n")
        handle.write(f"- `block_jacobi_ber_vs_snr.png`\n")
        handle.write(f"- `block_jacobi_se_vs_snr.png`\n")

    print("Generated files:")
    print(f"- {csv_path}")
    print(f"- {ber_png}")
    print(f"- {se_png}")
    print(f"- {report_path}")


def parse_args() -> SimConfig:
    parser = argparse.ArgumentParser(description="Simulate BER/SE for Block Jacobi deep-unfolding method")
    parser.add_argument("--nr", type=int, default=64)
    parser.add_argument("--nt", type=int, default=16)
    parser.add_argument("--n-sc", type=int, default=32)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--snr-db", type=str, default="0,5,10,15,20")
    parser.add_argument("--pilot-len", type=int, default=16)
    parser.add_argument("--pilot-snr-db", type=float, default=None)
    parser.add_argument("--modulation", choices=["16qam", "bpsk"], default="16qam")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bj-layers", type=int, default=12)
    parser.add_argument("--bj-block", type=int, default=4)
    parser.add_argument("--bj-adaptive-bounds", action="store_true")
    parser.add_argument("--out-dir", type=str, default="result_new/block Jacobi/simulation")

    args = parser.parse_args()
    snr_list = [float(token.strip()) for token in args.snr_db.split(",") if token.strip()]

    return SimConfig(
        nr=args.nr,
        nt=args.nt,
        n_sc=args.n_sc,
        batch=args.batch,
        trials=args.trials,
        snr_db_list=snr_list,
        pilot_len=args.pilot_len,
        pilot_snr_db=args.pilot_snr_db,
        modulation=args.modulation,
        seed=args.seed,
        bj_layers=args.bj_layers,
        bj_block=args.bj_block,
        bj_adaptive_bounds=args.bj_adaptive_bounds,
        out_dir=args.out_dir,
    )


def main() -> None:
    cfg = parse_args()
    metrics = run_simulation(cfg)
    save_outputs(cfg, metrics)


if __name__ == "__main__":
    main()
