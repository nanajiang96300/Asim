#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.evaluate_ldl_quality import (
    EvalConfig,
    bits_to_16qam,
    bits_to_64qam,
    demod_16qam,
    demod_64qam,
    estimate_se,
    ldl_inverse,
    ls_channel_estimate,
)

Array = np.ndarray


@dataclass
class CompareConfig:
    nr: int
    nt: int
    n_sc: int
    batch: int
    trials: int
    block_size: int
    snr_db_list: List[float]
    pilot_len: int
    pilot_snr_db: float | None
    modulation: str
    seed: int
    out_dir: str

    num_format: str
    reciprocal_mode: str
    trunc_mantissa_bits: int
    mac_chunk: int

    bj_layers: int
    bj_block: int
    bj_adaptive_bounds: bool
    bj_precond_solver: str
    bj_omega_policy: str
    bj_omega_tail_scale: float
    bj_corr_steps: int
    progress_every: int = 0
    watchdog_timeout_sec: float = 0.0
    max_seconds_per_snr: float = 0.0


def resolve_bj_layers(nt: int, requested_layers: int) -> int:
    if requested_layers > 0:
        return requested_layers
    if nt <= 16:
        return 8
    if nt <= 32:
        return 12
    return 80


def resolve_bj_block(nt: int, requested_block: int) -> int:
    if requested_block > 0:
        return requested_block
    if nt >= 64:
        return 16
    if nt >= 32:
        return 8
    return 4


def average_symbol_energy(constellation: Array) -> float:
    return float(np.mean(np.abs(constellation) ** 2))


def make_square_qam_constellation(order: int) -> Array:
    side = int(np.sqrt(order))
    if side * side != order:
        raise ValueError(f"Only square QAM is supported, got order={order}")
    levels = np.arange(-(side - 1), side, 2, dtype=np.float64)
    xv, yv = np.meshgrid(levels, levels)
    constellation = xv.reshape(-1) + 1j * yv.reshape(-1)
    constellation = constellation / np.sqrt(np.mean(np.abs(constellation) ** 2))
    return constellation.astype(np.complex128)


def generate_channel(rng: np.random.Generator, nr: int, nt: int) -> Array:
    return (rng.standard_normal((nr, nt)) + 1j * rng.standard_normal((nr, nt))) / np.sqrt(2.0)


def chebyshev_omega(n_layers: int, bmin: float = 0.1, bmax: float = 1.2) -> List[float]:
    omegas: List[float] = []
    for layer in range(n_layers):
        theta = np.pi * (2 * layer + 1) / (2 * n_layers)
        dt = 0.5 * (bmax + bmin) + 0.5 * (bmax - bmin) * np.cos(theta)
        omegas.append(float(1.0 / dt))
    return omegas


def chebyshev_omega_adaptive(b_mat: Array, n_layers: int, nt: int, floor: float = 1e-8) -> List[float]:
    eigvals = np.linalg.eigvals(b_mat)
    eigvals_real = np.real(eigvals)
    eigvals_real = eigvals_real[np.isfinite(eigvals_real)]
    eigvals_real = eigvals_real[eigvals_real > floor]

    if eigvals_real.size == 0:
        eigvals_real = np.array([floor, 1.0], dtype=np.float64)

    raw_min = float(np.min(eigvals_real))
    raw_max = float(np.max(eigvals_real))

    bmin = max(raw_min, 1e-2)
    bmax = max(raw_max, bmin * 2.0)

    return chebyshev_omega(n_layers=n_layers, bmin=bmin, bmax=bmax)


def stabilize_omegas(omegas: List[float], nt: int) -> List[float]:
    if nt >= 64:
        damp = 0.90
        omega_min = 0.20
        omega_max = 4.00
    else:
        damp = 1.00
        omega_min = 0.20
        omega_max = 4.00

    out: List[float] = []
    for omega in omegas:
        omega_damped = 1.0 + damp * (omega - 1.0)
        out.append(float(np.clip(omega_damped, omega_min, omega_max)))
    return out


def residual_optimal_omega(b_mat: Array, residual: Array, floor: float = 1e-12) -> float:
    br = b_mat @ residual
    denom = float(np.real(np.vdot(br, br)))
    if denom <= floor:
        return 1.0
    numer = float(np.real(np.vdot(residual, br)))
    omega = numer / denom
    return float(np.clip(omega, 0.05, 4.00))


def invert_block_ldl(block: Array, floor: float = 1e-12) -> Array:
    n = block.shape[0]
    l_mat = np.eye(n, dtype=np.complex128)
    d_vec = np.zeros(n, dtype=np.float64)

    for i in range(n):
        accum = 0.0
        for k in range(i):
            accum += (np.abs(l_mat[i, k]) ** 2) * d_vec[k]
        d_i = float(np.real(block[i, i]) - accum)
        d_i = max(d_i, floor)
        d_vec[i] = d_i

        for j in range(i + 1, n):
            term = 0.0 + 0.0j
            for k in range(i):
                term += l_mat[j, k] * np.conj(l_mat[i, k]) * d_vec[k]
            l_mat[j, i] = (block[j, i] - term) / d_i

    inv_l = np.linalg.inv(l_mat)
    inv_d = np.diag(1.0 / np.clip(d_vec, floor, None))
    return inv_l.conj().T @ inv_d @ inv_l


def invert_spd_block(block: Array, solver: str) -> Array:
    if solver == "direct" or solver == "direct2x2":
        if block.shape[0] != 2:
            return np.linalg.inv(block)
        if block.shape[0] == 2:
            a00 = block[0, 0]
            a01 = block[0, 1]
            a10 = block[1, 0]
            a11 = block[1, 1]
            det = a00 * a11 - a01 * a10
            det = det + (1e-12 + 0j)
            return np.array([[a11, -a01], [-a10, a00]], dtype=np.complex128) / det
        l_mat = np.linalg.cholesky(block)
        eye = np.eye(block.shape[0], dtype=np.complex128)
        y_mat = np.linalg.solve(l_mat, eye)
        return np.linalg.solve(l_mat.conj().T, y_mat)

    if solver == "cholesky":
        l_mat = np.linalg.cholesky(block)
        eye = np.eye(block.shape[0], dtype=np.complex128)
        y_mat = np.linalg.solve(l_mat, eye)
        return np.linalg.solve(l_mat.conj().T, y_mat)
    if solver == "ldl":
        try:
            return invert_block_ldl(block)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(block)

    eigvals, eigvecs = np.linalg.eigh(block)
    eigvals = np.clip(eigvals.real, 1e-12, None)
    return (eigvecs / eigvals[None, :]) @ eigvecs.conj().T


def build_block_richardson_preconditioner(a_mat: Array, blk: int = 4, precond_solver: str = "eig") -> Tuple[Array, Array]:
    n_stream = a_mat.shape[0]
    block_inv = np.zeros_like(a_mat, dtype=np.complex128)
    n_blk = n_stream // blk
    remainder = n_stream % blk

    for block_id in range(n_blk):
        start, stop = block_id * blk, (block_id + 1) * blk
        block = a_mat[start:stop, start:stop]
        block_inv[start:stop, start:stop] = invert_spd_block(block, solver=precond_solver)

    if remainder > 0:
        start = n_blk * blk
        block = a_mat[start:, start:]
        block_inv[start:, start:] = invert_spd_block(block, solver=precond_solver)

    b_mat = block_inv @ a_mat
    return b_mat, block_inv


def pcg_inverse(a_mat: Array, precond_inv: Array, max_iter: int = 15, tol: float = 1e-6) -> Array:
    n = a_mat.shape[0]
    x_mat = precond_inv.copy()
    identity = np.eye(n, dtype=np.complex128)
    r_mat = identity - a_mat @ x_mat
    z_mat = precond_inv @ r_mat
    p_mat = z_mat.copy()
    rz_old = np.real(np.sum(np.conj(r_mat) * z_mat, axis=0))

    for _ in range(max_iter):
        v_mat = a_mat @ p_mat
        p_ap = np.real(np.sum(np.conj(p_mat) * v_mat, axis=0))
        p_ap[p_ap < 1e-12] = 1e-12
        alpha = rz_old / p_ap

        x_mat = x_mat + p_mat * alpha[None, :]
        r_mat = r_mat - v_mat * alpha[None, :]

        if np.max(np.linalg.norm(r_mat, axis=0)) < tol:
            break

        z_mat = precond_inv @ r_mat
        rz_new = np.real(np.sum(np.conj(r_mat) * z_mat, axis=0))
        beta = rz_new / np.clip(rz_old, 1e-12, None)
        p_mat = z_mat + p_mat * beta[None, :]
        rz_old = rz_new

    return x_mat


def bj_chebyshev_inverse(
    a_mat: Array,
    n_layers: int,
    blk: int,
    adaptive_bounds: bool,
    precond_solver: str = "eig",
    omega_policy: str = "classic",
    omega_tail_scale: float = 1.0,
    corr_steps: int = 0,
) -> Array:
    nt = int(a_mat.shape[0])
    b_mat, b_inv = build_block_richardson_preconditioner(a_mat, blk=blk, precond_solver=precond_solver)
    identity = np.eye(a_mat.shape[0], dtype=np.complex128)
    y_mat = np.zeros_like(a_mat, dtype=np.complex128)
    omegas = chebyshev_omega_adaptive(b_mat, n_layers=n_layers, nt=nt) if adaptive_bounds else chebyshev_omega(
        n_layers=n_layers, bmin=0.1, bmax=1.2
    )

    if omega_policy == "hybrid64":
        omegas = stabilize_omegas(omegas, nt=nt)

    if omega_tail_scale < 1.0:
        half = max(len(omegas) // 2, 1)
        omegas = [w if idx < half else omega_tail_scale * w for idx, w in enumerate(omegas)]

    best_res_norm = np.inf
    for omega in omegas:
        residual = identity - b_mat @ y_mat
        res_norm = float(np.linalg.norm(residual))
        omega_eff = omega
        if omega_policy == "hybrid64" and nt >= 64:
            omega_ls = residual_optimal_omega(b_mat, residual)
            omega_eff = 0.55 * omega + 0.45 * omega_ls
            if res_norm > best_res_norm * 1.10:
                omega_eff = max(0.70 * omega_eff, 0.05)
        y_mat = y_mat + omega_eff * residual
        best_res_norm = min(best_res_norm, res_norm)

    for _ in range(max(corr_steps, 0)):
        residual = identity - b_mat @ y_mat
        alpha = residual_optimal_omega(b_mat, residual)
        y_mat = y_mat + alpha * residual

    return y_mat @ b_inv


def cholesky_inverse(a_mat: Array) -> Array:
    n = a_mat.shape[0]
    identity = np.eye(n, dtype=np.complex128)
    l_mat = np.linalg.cholesky(a_mat)
    y_mat = np.linalg.solve(l_mat, identity)
    x_mat = np.linalg.solve(l_mat.conj().T, y_mat)
    return x_mat


def run_compare(cfg: CompareConfig) -> List[Dict[str, float]]:
    rng = np.random.default_rng(cfg.seed)

    ldl_cfg = EvalConfig(
        nr=cfg.nr,
        nt=cfg.nt,
        n_sc=cfg.n_sc,
        batch=cfg.batch,
        trials=cfg.trials,
        block_size=cfg.block_size,
        snr_db_list=cfg.snr_db_list,
        channel_model="rayleigh",
        pilot_len=cfg.pilot_len,
        pilot_snr_db=cfg.pilot_snr_db,
        num_format=cfg.num_format,
        reciprocal_mode=cfg.reciprocal_mode,
        trunc_mantissa_bits=cfg.trunc_mantissa_bits,
        modulation=cfg.modulation,
        mac_chunk=cfg.mac_chunk,
        seed=cfg.seed,
        out_dir=cfg.out_dir,
    )

    if cfg.modulation == "16qam":
        constellation = make_square_qam_constellation(16)
    elif cfg.modulation == "64qam":
        constellation = make_square_qam_constellation(64)
    else:
        constellation = None
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

        ber_err_chol = 0
        ber_err_ldl = 0
        ber_err_bj = 0
        total_bits = 0

        se_sum_chol = 0.0
        se_sum_ldl = 0.0
        se_sum_bj = 0.0

        total_samples = cfg.trials * cfg.batch * cfg.n_sc
        last_progress_ts = time.time()
        snr_start_ts = last_progress_ts
        processed_samples = 0

        for sample_idx in range(total_samples):
            now = time.time()
            if cfg.max_seconds_per_snr > 0 and (now - snr_start_ts) > cfg.max_seconds_per_snr:
                print(
                    f"[watchdog] SNR {snr_db} dB 超过 {cfg.max_seconds_per_snr:.1f}s，提前结束该 SNR（已完成 {sample_idx}/{total_samples}）",
                    flush=True,
                )
                break

            if cfg.watchdog_timeout_sec > 0 and (now - last_progress_ts) > cfg.watchdog_timeout_sec:
                raise TimeoutError(
                    f"watchdog timeout: no progress for {cfg.watchdog_timeout_sec:.1f}s at SNR {snr_db} dB"
                )

            h_true = generate_channel(rng, cfg.nr, cfg.nt)
            h_est = ls_channel_estimate(rng, h_true, cfg.pilot_len, pilot_noise_var)

            if cfg.modulation == "16qam":
                tx_bits = rng.integers(0, 2, size=(cfg.nt, 4), dtype=np.int32)
                x_tx = bits_to_16qam(tx_bits)
                bits_per_sym = 4
            elif cfg.modulation == "64qam":
                tx_bits = rng.integers(0, 2, size=(cfg.nt, 6), dtype=np.int32)
                x_tx = bits_to_64qam(tx_bits)
                bits_per_sym = 6
            else:
                tx_bits = rng.integers(0, 2, size=(cfg.nt, 1), dtype=np.int32)
                x_tx = (2 * tx_bits[:, 0] - 1).astype(np.float64).astype(np.complex128)
                bits_per_sym = 1

            noise = np.sqrt(noise_var / 2.0) * (
                rng.standard_normal(cfg.nr) + 1j * rng.standard_normal(cfg.nr)
            )
            y_rx = h_true @ x_tx + noise

            a_est = h_est.conj().T @ h_est + (noise_var / symbol_energy) * np.eye(cfg.nt, dtype=np.complex128)
            h_h_est = h_est.conj().T

            a_inv_chol = cholesky_inverse(a_est)
            a_inv_ldl, _ = ldl_inverse(a_est, ldl_cfg, block_size=cfg.block_size)
            _, b_inv = build_block_richardson_preconditioner(
                a_est,
                blk=cfg.bj_block,
                precond_solver=cfg.bj_precond_solver,
            )
            a_inv_bj = pcg_inverse(
                a_est,
                b_inv,
                max_iter=max(cfg.bj_layers, 1),
                tol=1e-6,
            )

            w_chol = a_inv_chol @ h_h_est
            w_ldl = a_inv_ldl @ h_h_est
            w_bj = a_inv_bj @ h_h_est

            xhat_chol = w_chol @ y_rx
            xhat_ldl = w_ldl @ y_rx
            xhat_bj = w_bj @ y_rx

            if cfg.modulation == "16qam":
                bits_hat_chol = demod_16qam(xhat_chol)
                bits_hat_ldl = demod_16qam(xhat_ldl)
                bits_hat_bj = demod_16qam(xhat_bj)
            elif cfg.modulation == "64qam":
                bits_hat_chol = demod_64qam(xhat_chol)
                bits_hat_ldl = demod_64qam(xhat_ldl)
                bits_hat_bj = demod_64qam(xhat_bj)
            else:
                bits_hat_chol = (np.real(xhat_chol) >= 0).astype(np.int32)[:, None]
                bits_hat_ldl = (np.real(xhat_ldl) >= 0).astype(np.int32)[:, None]
                bits_hat_bj = (np.real(xhat_bj) >= 0).astype(np.int32)[:, None]

            ber_err_chol += int(np.sum(bits_hat_chol != tx_bits))
            ber_err_ldl += int(np.sum(bits_hat_ldl != tx_bits))
            ber_err_bj += int(np.sum(bits_hat_bj != tx_bits))
            total_bits += cfg.nt * bits_per_sym

            se_sum_chol += estimate_se(w_chol, h_true, noise_var, cfg.modulation)
            se_sum_ldl += estimate_se(w_ldl, h_true, noise_var, cfg.modulation)
            se_sum_bj += estimate_se(w_bj, h_true, noise_var, cfg.modulation)
            processed_samples += 1

            last_progress_ts = time.time()
            if cfg.progress_every > 0 and ((sample_idx + 1) % cfg.progress_every == 0 or (sample_idx + 1) == total_samples):
                elapsed = last_progress_ts - snr_start_ts
                print(
                    f"[heartbeat] SNR {snr_db} dB: {sample_idx + 1}/{total_samples} samples, elapsed {elapsed:.2f}s",
                    flush=True,
                )

        ber_chol = ber_err_chol / max(total_bits, 1)
        ber_ldl = ber_err_ldl / max(total_bits, 1)
        ber_bj = ber_err_bj / max(total_bits, 1)

        effective_samples = max(processed_samples, 1)
        se_chol = se_sum_chol / effective_samples
        se_ldl = se_sum_ldl / effective_samples
        se_bj = se_sum_bj / effective_samples

        metrics.append(
            {
                "snr_db": snr_db,
                "ber_cholesky": ber_chol,
                "ber_ldl": ber_ldl,
                "ber_bj_iterative": ber_bj,
                "ber_bj_deepunfold": ber_bj,
                "se_cholesky": se_chol,
                "se_ldl": se_ldl,
                "se_bj_iterative": se_bj,
                "se_bj_deepunfold": se_bj,
                "ber_gap_bj_vs_chol": ber_bj - ber_chol,
                "se_gap_bj_vs_chol": se_bj - se_chol,
            }
        )

        print(
            f"[progress] SNR {snr_db} dB done in {time.time() - t_snr:.2f}s, elapsed {time.time() - t_global:.2f}s",
            flush=True,
        )

    return metrics


def save_outputs(cfg: CompareConfig, metrics: List[Dict[str, float]]) -> Tuple[str, str, str, str]:
    os.makedirs(cfg.out_dir, exist_ok=True)

    csv_path = os.path.join(cfg.out_dir, "bj_iterative_vs_cholesky_ldl_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "snr_db",
                "ber_cholesky",
                "ber_ldl",
                "ber_bj_iterative",
                "ber_bj_deepunfold",
                "se_cholesky",
                "se_ldl",
                "se_bj_iterative",
                "se_bj_deepunfold",
                "ber_gap_bj_vs_chol",
                "se_gap_bj_vs_chol",
            ],
        )
        writer.writeheader()
        writer.writerows(metrics)

    snr = [r["snr_db"] for r in metrics]
    ber_chol = [r["ber_cholesky"] for r in metrics]
    ber_ldl = [r["ber_ldl"] for r in metrics]
    ber_bj = [r["ber_bj_iterative"] for r in metrics]

    se_chol = [r["se_cholesky"] for r in metrics]
    se_ldl = [r["se_ldl"] for r in metrics]
    se_bj = [r["se_bj_iterative"] for r in metrics]

    eps = 1e-8
    ber_chol_plot = [max(v, eps) for v in ber_chol]
    ber_ldl_plot = [max(v, eps) for v in ber_ldl]
    ber_bj_plot = [max(v, eps) for v in ber_bj]

    ber_png = os.path.join(cfg.out_dir, "ber_vs_snr_cholesky_ldl_bj_iterative.png")
    plt.figure(figsize=(7.4, 5.2))
    plt.semilogy(snr, ber_chol_plot, marker="o", label="Cholesky-MMSE")
    plt.semilogy(snr, ber_ldl_plot, marker="s", linestyle="--", label="Block-LDL")
    plt.semilogy(snr, ber_bj_plot, marker="^", linestyle="-.", label="BJ-Iterative")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("BER vs SNR (Same Environment)")
    plt.grid(True, which="both", linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ber_png, dpi=220)
    plt.close()

    se_png = os.path.join(cfg.out_dir, "se_vs_snr_cholesky_ldl_bj_iterative.png")
    plt.figure(figsize=(7.4, 5.2))
    plt.plot(snr, se_chol, marker="o", label="Cholesky-MMSE")
    plt.plot(snr, se_ldl, marker="s", linestyle="--", label="Block-LDL")
    plt.plot(snr, se_bj, marker="^", linestyle="-.", label="BJ-Iterative")
    plt.xlabel("SNR (dB)")
    plt.ylabel("SE (bits/s/Hz)")
    plt.title("SE vs SNR (Same Environment)")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(se_png, dpi=220)
    plt.close()

    combo_png = os.path.join(cfg.out_dir, "se_ber_overlay_cholesky_ldl_bj_iterative.png")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    axes[0].semilogy(snr, ber_chol_plot, marker="o", label="Cholesky-MMSE")
    axes[0].semilogy(snr, ber_ldl_plot, marker="s", linestyle="--", label="Block-LDL")
    axes[0].semilogy(snr, ber_bj_plot, marker="^", linestyle="-.", label="BJ-Iterative")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("BER")
    axes[0].set_title("BER")
    axes[0].grid(True, which="both", linestyle=":", alpha=0.7)

    axes[1].plot(snr, se_chol, marker="o", label="Cholesky-MMSE")
    axes[1].plot(snr, se_ldl, marker="s", linestyle="--", label="Block-LDL")
    axes[1].plot(snr, se_bj, marker="^", linestyle="-.", label="BJ-Iterative")
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("SE (bits/s/Hz)")
    axes[1].set_title("SE")
    axes[1].grid(True, linestyle=":", alpha=0.7)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Cholesky vs LDL vs BJ-Iterative", y=1.02)
    fig.tight_layout()
    fig.savefig(combo_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    report_path = os.path.join(cfg.out_dir, "bj_iterative_vs_cholesky_ldl_report.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# BJ-Iterative 与 Cholesky/LDL 对比报告\n\n")
        handle.write("## 方法\n")
        handle.write("- Cholesky-MMSE：对 `A=H^H H + (sigma^2/E_s)I` 做 Cholesky 分解求逆。\n")
        handle.write("- Block-LDL：沿用项目中块 LDL 近似求逆流程。\n")
        handle.write("- BJ-Iterative：Block-Jacobi 预条件 + PCG（预条件共轭梯度）并行迭代求逆。\n\n")
        handle.write("## BJ 参数\n")
        handle.write(f"- layers: `{cfg.bj_layers}`\n")
        handle.write(f"- block size: `{cfg.bj_block}`\n")
        handle.write(f"- preconditioner solver: `{cfg.bj_precond_solver}`\n")
        handle.write(f"- omega policy: `{cfg.bj_omega_policy}`\n")
        handle.write(f"- omega tail scale: `{cfg.bj_omega_tail_scale}`\n")
        handle.write(f"- residual correction steps: `{cfg.bj_corr_steps}`\n")
        handle.write(f"- adaptive bounds: `{cfg.bj_adaptive_bounds}`\n\n")
        handle.write("## 统一环境\n")
        handle.write(
            f"- nr={cfg.nr}, nt={cfg.nt}, n_sc={cfg.n_sc}, batch={cfg.batch}, trials={cfg.trials}, snr={cfg.snr_db_list}\n"
        )
        handle.write(
            f"- modulation={cfg.modulation}, pilot_len={cfg.pilot_len}, num_format={cfg.num_format}, seed={cfg.seed}\n\n"
        )
        handle.write("## 文件\n")
        handle.write(f"- metrics: `{os.path.basename(csv_path)}`\n")
        handle.write(f"- BER: `{os.path.basename(ber_png)}`\n")
        handle.write(f"- SE: `{os.path.basename(se_png)}`\n")
        handle.write(f"- overlay: `{os.path.basename(combo_png)}`\n")

    return csv_path, ber_png, se_png, combo_png


def parse_args() -> CompareConfig:
    parser = argparse.ArgumentParser(description="Evaluate BJ-Iterative vs Cholesky and LDL under same setup.")

    parser.add_argument("--nr", type=int, default=64)
    parser.add_argument("--nt", type=int, default=16)
    parser.add_argument("--n-sc", type=int, default=168)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--snr-db", type=str, default="0,5,10,15,20,25,30")
    parser.add_argument("--pilot-len", type=int, default=16)
    parser.add_argument("--pilot-snr-db", type=float, default=None)
    parser.add_argument("--modulation", type=str, default="64qam", choices=["16qam", "64qam", "bpsk"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="results/LDL/bj_iterative_compare")

    parser.add_argument("--num-format", type=str, default="fp16", choices=["fp16", "fp64"])
    parser.add_argument("--reciprocal-mode", type=str, default="approx", choices=["approx", "exact"])
    parser.add_argument("--trunc-mantissa-bits", type=int, default=8)
    parser.add_argument("--mac-chunk", type=int, default=4)

    parser.add_argument("--bj-layers", type=int, default=0, help="<=0 means auto by nt: 16->8, 32->12, 64->80")
    parser.add_argument("--bj-block", type=int, default=0, help="<=0 means auto by nt: 16->4, 32->8, 64->16")
    parser.add_argument("--bj-adaptive-bounds", action="store_true")
    parser.add_argument("--bj-precond-solver", type=str, default="eig", choices=["eig", "cholesky", "ldl"])
    parser.add_argument("--bj-omega-policy", type=str, default="classic", choices=["classic", "hybrid64"])
    parser.add_argument("--bj-omega-tail-scale", type=float, default=1.0)
    parser.add_argument("--bj-corr-steps", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=0, help=">0 prints heartbeat every N samples")
    parser.add_argument("--watchdog-timeout-sec", type=float, default=0.0, help=">0 raises timeout if no progress for N seconds")
    parser.add_argument("--max-seconds-per-snr", type=float, default=0.0, help=">0 aborts current SNR once elapsed time exceeds N seconds")

    args = parser.parse_args()
    snr_list = [float(token.strip()) for token in args.snr_db.split(",") if token.strip()]

    return CompareConfig(
        nr=args.nr,
        nt=args.nt,
        n_sc=args.n_sc,
        batch=args.batch,
        trials=args.trials,
        block_size=args.block_size,
        snr_db_list=snr_list,
        pilot_len=args.pilot_len,
        pilot_snr_db=args.pilot_snr_db,
        modulation=args.modulation,
        seed=args.seed,
        out_dir=args.out_dir,
        num_format=args.num_format,
        reciprocal_mode=args.reciprocal_mode,
        trunc_mantissa_bits=args.trunc_mantissa_bits,
        mac_chunk=args.mac_chunk,
        bj_layers=args.bj_layers,
        bj_block=args.bj_block,
        bj_adaptive_bounds=args.bj_adaptive_bounds,
        bj_precond_solver=args.bj_precond_solver,
        bj_omega_policy=args.bj_omega_policy,
        bj_omega_tail_scale=args.bj_omega_tail_scale,
        bj_corr_steps=args.bj_corr_steps,
        progress_every=args.progress_every,
        watchdog_timeout_sec=args.watchdog_timeout_sec,
        max_seconds_per_snr=args.max_seconds_per_snr,
    )


def main() -> None:
    cfg = parse_args()
    if cfg.nt % cfg.block_size != 0:
        raise SystemExit(f"nt ({cfg.nt}) must be divisible by block_size ({cfg.block_size})")

    cfg.bj_layers = resolve_bj_layers(cfg.nt, cfg.bj_layers)
    cfg.bj_block = resolve_bj_block(cfg.nt, cfg.bj_block)

    metrics = run_compare(cfg)
    outputs = save_outputs(cfg, metrics)

    print("BJ-Iterative comparison finished.")
    print("Generated files:")
    for path in outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()
