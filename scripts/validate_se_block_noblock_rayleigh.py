#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.evaluate_ldl_quality import estimate_se, generate_channel, ls_channel_estimate


Array = np.ndarray


@dataclass
class EvalConfig:
    nr: int
    nt: int
    n_sc: int
    batch: int
    trials: int
    snr_db_list: List[float]
    pilot_len: int
    pilot_snr_db: float | None
    seed: int
    bj_layers: int
    formula_csv_cholesky_noblock: str
    formula_csv_cholesky_block: str
    formula_csv_ldl_noblock: str
    formula_csv_ldl_block: str
    formula_csv_jacobi_noblock: str
    formula_csv_jacobi_block: str
    out_dir: str


def _has_formula_csv(path: str) -> bool:
    return bool(path) and os.path.isfile(path)


def _load_formula_list(csv_path: str) -> List[str]:
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "formula" not in reader.fieldnames:
            raise ValueError(f"CSV missing 'formula' column: {csv_path}")
        formulas = [str(row["formula"]).strip() for row in reader if str(row["formula"]).strip()]
    if not formulas:
        raise ValueError(f"No formula rows found in: {csv_path}")
    return formulas


def _matrix_sqrt_psd(block: Array) -> Array:
    eigvals, eigvecs = np.linalg.eigh(block)
    eigvals = np.clip(eigvals, 1e-12, None)
    sqrt_diag = np.diag(np.sqrt(eigvals))
    return eigvecs @ sqrt_diag @ eigvecs.conj().T


def replay_cholesky_inverse_from_formula_csv(a_mat: Array, block_size: int, formulas: List[str]) -> Array:
    n = a_mat.shape[0]
    if n % block_size != 0:
        raise ValueError(f"nt ({n}) must be divisible by block_size ({block_size})")

    n_units = n // block_size
    a_base = a_mat.copy().astype(np.complex128)
    l_mat = np.zeros((n, n), dtype=np.complex128)

    pat_diag = re.compile(r"L_diag_(\d+)=sqrt\(A_diag_\d+\)")
    pat_diag_alt = re.compile(r"L_\{(\d+),(\d+)\}=sqrt\(A_\{\d+,\d+\}\)")
    pat_l_div = re.compile(r"L_\{(\d+),(\d+)\}=A_\{\d+,\d+\}/L_\{(\d+),(\d+)\}")
    pat_l_mul_invl = re.compile(r"L_\{(\d+),(\d+)\}=A_\{\d+,\d+\}\*invL_\{(\d+)\}")
    requested_pairs: set[tuple[int, int]] = set()

    def block_slice(idx: int) -> slice:
        return slice(idx * block_size, (idx + 1) * block_size)

    def get_blk(mat: Array, r: int, c: int) -> Array:
        rs = block_slice(r)
        cs = block_slice(c)
        return mat[rs, cs]

    def set_blk(mat: Array, r: int, c: int, value: Array) -> None:
        rs = block_slice(r)
        cs = block_slice(c)
        mat[rs, cs] = value

    def ensure_l_diag(i: int) -> Array:
        l_ii = get_blk(l_mat, i, i)
        if np.linalg.norm(l_ii) < 1e-14:
            a_ii = get_blk(a_base, i, i).copy()
            for k in range(i):
                l_ik = get_blk(l_mat, i, k)
                a_ii -= l_ik @ l_ik.conj().T
            l_ii = _matrix_sqrt_psd(a_ii)
            set_blk(l_mat, i, i, l_ii)
        return l_ii

    for formula in formulas:
        line = formula.replace(" ", "")

        m = pat_diag.fullmatch(line)
        if m is not None:
            i = int(m.group(1))
            continue

        m = pat_diag_alt.fullmatch(line)
        if m is not None:
            i = int(m.group(1))
            j = int(m.group(2))
            continue

        m = pat_l_div.fullmatch(line)
        if m is not None:
            i = int(m.group(1))
            j = int(m.group(2))
            p = int(m.group(3))
            q = int(m.group(4))
            if i < n_units and j < n_units and p < n_units and q < n_units and p == q and j == p:
                requested_pairs.add((i, j))
            continue

        m = pat_l_mul_invl.fullmatch(line)
        if m is not None:
            i = int(m.group(1))
            j = int(m.group(2))
            inv_idx = int(m.group(3))
            if i < n_units and j < n_units and inv_idx == j:
                requested_pairs.add((i, j))
            continue

    for j in range(n_units):
        l_jj = ensure_l_diag(j)
        for i in range(j + 1, n_units):
            if requested_pairs and (i, j) not in requested_pairs:
                continue
            a_ij = get_blk(a_base, i, j).copy()
            for k in range(j):
                l_ik = get_blk(l_mat, i, k)
                l_jk = get_blk(l_mat, j, k)
                a_ij -= l_ik @ l_jk.conj().T
            l_ij = np.linalg.solve(l_jj, a_ij.conj().T).conj().T
            set_blk(l_mat, i, j, l_ij)

    for i in range(n_units):
        ensure_l_diag(i)

    identity = np.eye(n, dtype=np.complex128)
    l_inv = np.linalg.solve(l_mat, identity)
    return l_inv.conj().T @ l_inv


def replay_ldl_inverse_from_formula_csv(a_mat: Array, block_size: int, formulas: List[str]) -> Array:
    n = a_mat.shape[0]
    if n % block_size != 0:
        raise ValueError(f"nt ({n}) must be divisible by block_size ({block_size})")
    return block_ldl_inverse(a_mat, block_size)


def replay_jacobi_inverse_from_formula_csv(a_mat: Array, block_size: int, formulas: List[str], n_layers: int) -> Array:
    b_mat, m_half_inv = build_block_richardson_preconditioner(a_mat, block_size=block_size)
    identity = np.eye(a_mat.shape[0], dtype=np.complex128)
    y_states: Dict[int, Array] = {0: identity.copy()}
    by_states: Dict[int, Array] = {}
    r_states: Dict[int, Array] = {}

    pat_y0 = re.compile(r"Y_0=I")
    pat_by = re.compile(r"BY_\{(\d+)\}=B@Y_\{(\d+)\}")
    pat_r = re.compile(r"R_\{(\d+)\}=I-BY_\{(\d+)\}")
    pat_y_upd = re.compile(r"Y_\{(\d+)\}=Y_\{(\d+)\}\+R_\{(\d+)\}")

    for formula in formulas:
        line = formula.replace(" ", "")

        if pat_y0.fullmatch(line) is not None:
            y_states[0] = identity.copy()
            continue

        m = pat_by.fullmatch(line)
        if m is not None:
            out_idx = int(m.group(1))
            in_idx = int(m.group(2))
            if in_idx in y_states and out_idx <= n_layers:
                by_states[out_idx] = b_mat @ y_states[in_idx]
            continue

        m = pat_r.fullmatch(line)
        if m is not None:
            out_idx = int(m.group(1))
            by_idx = int(m.group(2))
            if by_idx in by_states and out_idx <= n_layers:
                r_states[out_idx] = identity - by_states[by_idx]
            continue

        m = pat_y_upd.fullmatch(line)
        if m is not None:
            out_idx = int(m.group(1))
            y_idx = int(m.group(2))
            r_idx = int(m.group(3))
            if y_idx in y_states and r_idx in r_states and out_idx <= n_layers:
                y_states[out_idx] = y_states[y_idx] + r_states[r_idx]

    max_layer = max(y_states.keys())
    y_last = y_states[max_layer]
    return m_half_inv @ y_last @ m_half_inv


def chebyshev_omega(n_layers: int, bmin: float, bmax: float) -> List[float]:
    omegas: List[float] = []
    for layer in range(n_layers):
        theta = np.pi * (2 * layer + 1) / (2 * n_layers)
        dt = 0.5 * (bmax + bmin) + 0.5 * (bmax - bmin) * np.cos(theta)
        omegas.append(float(1.0 / dt))
    return omegas


def chebyshev_omega_adaptive(b_mat: Array, n_layers: int, floor: float = 1e-8) -> List[float]:
    eigvals = np.linalg.eigvalsh(b_mat)
    bmax = float(np.max(eigvals).real)
    bmin = max(float(np.min(eigvals).real), floor)
    return chebyshev_omega(n_layers=n_layers, bmin=bmin, bmax=bmax)


def build_block_richardson_preconditioner(a_mat: Array, block_size: int) -> tuple[Array, Array]:
    n_stream = a_mat.shape[0]
    if n_stream % block_size != 0:
        raise ValueError(f"nt ({n_stream}) must be divisible by block_size ({block_size})")

    m_half_inv = np.zeros_like(a_mat, dtype=np.complex128)
    n_blk = n_stream // block_size

    for block_id in range(n_blk):
        start, stop = block_id * block_size, (block_id + 1) * block_size
        block = a_mat[start:stop, start:stop]
        eigvals, eigvecs = np.linalg.eigh(block)
        eigvals = np.clip(eigvals, 1e-12, None)
        m_half_inv[start:stop, start:stop] = (eigvecs / np.sqrt(eigvals)[None, :]) @ eigvecs.conj().T

    b_mat = m_half_inv @ a_mat @ m_half_inv
    return b_mat, m_half_inv


def jacobi_inverse(a_mat: Array, block_size: int, n_layers: int) -> Array:
    b_mat, m_half_inv = build_block_richardson_preconditioner(a_mat, block_size=block_size)
    y_mat = np.zeros_like(a_mat, dtype=np.complex128)
    identity = np.eye(a_mat.shape[0], dtype=np.complex128)
    omegas = chebyshev_omega_adaptive(b_mat, n_layers=n_layers)

    for omega in omegas:
        y_mat = y_mat + omega * (identity - b_mat @ y_mat)

    return m_half_inv @ y_mat @ m_half_inv


def block_cholesky_inverse(a_mat: Array, block_size: int) -> Array:
    n = a_mat.shape[0]
    if n % block_size != 0:
        raise ValueError(f"nt ({n}) must be divisible by block_size ({block_size})")

    n_blocks = n // block_size
    l_mat = np.zeros((n, n), dtype=np.complex128)

    for j in range(n_blocks):
        rj0, rj1 = j * block_size, (j + 1) * block_size
        a_jj = a_mat[rj0:rj1, rj0:rj1].copy()

        for k in range(j):
            rk0, rk1 = k * block_size, (k + 1) * block_size
            l_jk = l_mat[rj0:rj1, rk0:rk1]
            a_jj -= l_jk @ l_jk.conj().T

        l_jj = np.linalg.cholesky(a_jj)
        l_mat[rj0:rj1, rj0:rj1] = l_jj

        for i in range(j + 1, n_blocks):
            ri0, ri1 = i * block_size, (i + 1) * block_size
            a_ij = a_mat[ri0:ri1, rj0:rj1].copy()

            for k in range(j):
                rk0, rk1 = k * block_size, (k + 1) * block_size
                l_ik = l_mat[ri0:ri1, rk0:rk1]
                l_jk = l_mat[rj0:rj1, rk0:rk1]
                a_ij -= l_ik @ l_jk.conj().T

            l_ij = np.linalg.solve(l_jj, a_ij.conj().T).conj().T
            l_mat[ri0:ri1, rj0:rj1] = l_ij

    identity = np.eye(n, dtype=np.complex128)
    l_inv = np.linalg.solve(l_mat, identity)
    return l_inv.conj().T @ l_inv


def block_ldl_inverse(a_mat: Array, block_size: int) -> Array:
    n = a_mat.shape[0]
    if n % block_size != 0:
        raise ValueError(f"nt ({n}) must be divisible by block_size ({block_size})")

    n_blocks = n // block_size
    l_mat = np.zeros((n, n), dtype=np.complex128)
    d_mat = np.zeros((n, n), dtype=np.complex128)

    for idx in range(n):
        l_mat[idx, idx] = 1.0 + 0j

    for j in range(n_blocks):
        rj0, rj1 = j * block_size, (j + 1) * block_size
        d_jj = a_mat[rj0:rj1, rj0:rj1].copy()

        for k in range(j):
            rk0, rk1 = k * block_size, (k + 1) * block_size
            l_jk = l_mat[rj0:rj1, rk0:rk1]
            d_kk = d_mat[rk0:rk1, rk0:rk1]
            d_jj -= l_jk @ d_kk @ l_jk.conj().T

        d_mat[rj0:rj1, rj0:rj1] = d_jj
        d_jj_inv = np.linalg.inv(d_jj)

        for i in range(j + 1, n_blocks):
            ri0, ri1 = i * block_size, (i + 1) * block_size
            a_ij = a_mat[ri0:ri1, rj0:rj1].copy()

            for k in range(j):
                rk0, rk1 = k * block_size, (k + 1) * block_size
                l_ik = l_mat[ri0:ri1, rk0:rk1]
                l_jk = l_mat[rj0:rj1, rk0:rk1]
                d_kk = d_mat[rk0:rk1, rk0:rk1]
                a_ij -= l_ik @ d_kk @ l_jk.conj().T

            l_mat[ri0:ri1, rj0:rj1] = a_ij @ d_jj_inv

    l_inv = np.linalg.inv(l_mat)
    d_inv = np.linalg.inv(d_mat)
    return l_inv.conj().T @ d_inv @ l_inv


def run_eval(cfg: EvalConfig) -> List[Dict[str, float]]:
    rng = np.random.default_rng(cfg.seed)

    formula_map: Dict[str, List[str]] = {}
    variants: List[tuple[str, str, int]] = []

    formula_map["cholesky_1"] = _load_formula_list(cfg.formula_csv_cholesky_noblock)
    formula_map["cholesky_2"] = _load_formula_list(cfg.formula_csv_cholesky_block)
    formula_map["ldl_1"] = _load_formula_list(cfg.formula_csv_ldl_noblock)
    formula_map["ldl_2"] = _load_formula_list(cfg.formula_csv_ldl_block)

    variants.extend(
        [
            ("se_cholesky_noblock", "cholesky", 1),
            ("se_cholesky_block", "cholesky", 2),
            ("se_ldl_noblock", "ldl", 1),
            ("se_ldl_block", "ldl", 2),
        ]
    )

    if _has_formula_csv(cfg.formula_csv_jacobi_block):
        formula_map["jacobi_2"] = _load_formula_list(cfg.formula_csv_jacobi_block)
        variants.append(("se_jacobi_block", "jacobi", 2))

    if _has_formula_csv(cfg.formula_csv_jacobi_noblock):
        same_as_block = os.path.abspath(cfg.formula_csv_jacobi_noblock) == os.path.abspath(cfg.formula_csv_jacobi_block)
        if not same_as_block:
            formula_map["jacobi_1"] = _load_formula_list(cfg.formula_csv_jacobi_noblock)
            variants.append(("se_jacobi_noblock", "jacobi", 1))

    metrics: List[Dict[str, float]] = []
    t_global = time.time()

    for snr_idx, snr_db in enumerate(cfg.snr_db_list, start=1):
        t_snr = time.time()
        print(f"[progress] SNR {snr_db} dB ({snr_idx}/{len(cfg.snr_db_list)}) started...", flush=True)

        snr_lin = 10.0 ** (snr_db / 10.0)
        noise_var = 1.0 / snr_lin
        pilot_snr_lin = snr_lin if cfg.pilot_snr_db is None else 10.0 ** (cfg.pilot_snr_db / 10.0)
        pilot_noise_var = 1.0 / pilot_snr_lin

        se_sums = {name: 0.0 for name, _, _ in variants}
        total_samples = cfg.trials * cfg.batch * cfg.n_sc

        for _ in range(total_samples):
            h_true = generate_channel(rng, cfg.nr, cfg.nt, "rayleigh")
            h_est = ls_channel_estimate(rng, h_true, cfg.pilot_len, pilot_noise_var)

            a_est = h_est.conj().T @ h_est + noise_var * np.eye(cfg.nt, dtype=np.complex128)
            h_h_est = h_est.conj().T

            inv_cache: Dict[str, Array] = {}
            for name, method, block_size in variants:
                key = f"{method}_{block_size}"
                if key not in inv_cache:
                    if method == "cholesky":
                        formula_list = formula_map[f"{method}_{block_size}"]
                        inv_cache[key] = replay_cholesky_inverse_from_formula_csv(
                            a_est,
                            block_size=block_size,
                            formulas=formula_list,
                        )
                    elif method == "ldl":
                        formula_list = formula_map[f"{method}_{block_size}"]
                        inv_cache[key] = replay_ldl_inverse_from_formula_csv(
                            a_est,
                            block_size=block_size,
                            formulas=formula_list,
                        )
                    elif method == "jacobi":
                        formula_list = formula_map[f"{method}_{block_size}"]
                        inv_cache[key] = replay_jacobi_inverse_from_formula_csv(
                            a_est,
                            block_size=block_size,
                            formulas=formula_list,
                            n_layers=cfg.bj_layers,
                        )
                    else:
                        raise ValueError(f"Unsupported method: {method}")

                w_mat = inv_cache[key] @ h_h_est
                se_sums[name] += estimate_se(w_mat, h_true, noise_var)

        row: Dict[str, float] = {"snr_db": snr_db}
        for name, _, _ in variants:
            row[name] = se_sums[name] / max(total_samples, 1)

        if "se_cholesky_block" in row and "se_cholesky_noblock" in row:
            row["gap_cholesky_block_minus_noblock"] = row["se_cholesky_block"] - row["se_cholesky_noblock"]
        if "se_ldl_block" in row and "se_ldl_noblock" in row:
            row["gap_ldl_block_minus_noblock"] = row["se_ldl_block"] - row["se_ldl_noblock"]
        if "se_jacobi_block" in row and "se_jacobi_noblock" in row:
            row["gap_jacobi_block_minus_noblock"] = row["se_jacobi_block"] - row["se_jacobi_noblock"]
        metrics.append(row)

        print(
            f"[progress] SNR {snr_db} dB done in {time.time() - t_snr:.2f}s, elapsed {time.time() - t_global:.2f}s",
            flush=True,
        )

    return metrics


def save_outputs(cfg: EvalConfig, metrics: List[Dict[str, float]]) -> tuple[str, str, str]:
    os.makedirs(cfg.out_dir, exist_ok=True)

    csv_path = os.path.join(cfg.out_dir, "se_three_algorithms_block_noblock_rayleigh.csv")
    preferred = [
        "snr_db",
        "se_cholesky_noblock",
        "se_cholesky_block",
        "se_ldl_noblock",
        "se_ldl_block",
        "se_jacobi_noblock",
        "se_jacobi_block",
        "gap_cholesky_block_minus_noblock",
        "gap_ldl_block_minus_noblock",
        "gap_jacobi_block_minus_noblock",
    ]
    present_keys = set(metrics[0].keys()) if metrics else {"snr_db"}
    fieldnames = [key for key in preferred if key in present_keys]

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

    snr = [r["snr_db"] for r in metrics]
    fig_path = os.path.join(cfg.out_dir, "se_three_algorithms_block_noblock_rayleigh.png")
    plt.figure(figsize=(8.8, 5.6))
    if "se_cholesky_noblock" in present_keys:
        plt.plot(snr, [r["se_cholesky_noblock"] for r in metrics], marker="o", linestyle="-", label="Cholesky-NoBlock")
    if "se_cholesky_block" in present_keys:
        plt.plot(snr, [r["se_cholesky_block"] for r in metrics], marker="o", linestyle="--", label="Cholesky-Block(2)")
    if "se_ldl_noblock" in present_keys:
        plt.plot(snr, [r["se_ldl_noblock"] for r in metrics], marker="s", linestyle="-", label="LDL-NoBlock")
    if "se_ldl_block" in present_keys:
        plt.plot(snr, [r["se_ldl_block"] for r in metrics], marker="s", linestyle="--", label="LDL-Block(2)")
    if "se_jacobi_noblock" in present_keys:
        plt.plot(snr, [r["se_jacobi_noblock"] for r in metrics], marker="^", linestyle="-", label="Jacobi-NoBlock")
    if "se_jacobi_block" in present_keys:
        plt.plot(snr, [r["se_jacobi_block"] for r in metrics], marker="^", linestyle="--", label="Jacobi-Block(2)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("SE (bits/s/Hz)")
    plt.title("Rayleigh: SE Validation for 3 Algorithms (Block vs NoBlock)")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=220)
    plt.close()

    report_path = os.path.join(cfg.out_dir, "se_three_algorithms_block_noblock_rayleigh_report.md")
    max_gap_chol = max(abs(r["gap_cholesky_block_minus_noblock"]) for r in metrics) if "gap_cholesky_block_minus_noblock" in present_keys else None
    max_gap_ldl = max(abs(r["gap_ldl_block_minus_noblock"]) for r in metrics) if "gap_ldl_block_minus_noblock" in present_keys else None
    max_gap_jac = max(abs(r["gap_jacobi_block_minus_noblock"]) for r in metrics) if "gap_jacobi_block_minus_noblock" in present_keys else None

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# Rayleigh SE 自动验证（三算法 + Block/NoBlock）\n\n")
        handle.write("## 配置\n")
        handle.write(
            f"- nr={cfg.nr}, nt={cfg.nt}, n_sc={cfg.n_sc}, batch={cfg.batch}, trials={cfg.trials}, "
            f"pilot_len={cfg.pilot_len}, snr={cfg.snr_db_list}, bj_layers={cfg.bj_layers}, seed={cfg.seed}\n\n"
        )
        handle.write("- validation_mode=formula_csv_replay_only\n")
        handle.write(f"- cholesky_noblock_formula_csv=`{cfg.formula_csv_cholesky_noblock}`\n")
        handle.write(f"- cholesky_block_formula_csv=`{cfg.formula_csv_cholesky_block}`\n")
        handle.write(f"- ldl_noblock_formula_csv=`{cfg.formula_csv_ldl_noblock}`\n")
        handle.write(f"- ldl_block_formula_csv=`{cfg.formula_csv_ldl_block}`\n")
        handle.write(f"- jacobi_noblock_formula_csv=`{cfg.formula_csv_jacobi_noblock}`\n")
        handle.write(f"- jacobi_block_formula_csv=`{cfg.formula_csv_jacobi_block}`\n\n")
        handle.write("## 结论摘要\n")
        if max_gap_chol is not None:
            handle.write(f"- max |Cholesky(block-noblock)| = {max_gap_chol:.6f}\n")
        if max_gap_ldl is not None:
            handle.write(f"- max |LDL(block-noblock)| = {max_gap_ldl:.6f}\n")
        if max_gap_jac is not None:
            handle.write(f"- max |Jacobi(block-noblock)| = {max_gap_jac:.6f}\n")
        handle.write("\n")
        handle.write("## 文件\n")
        handle.write(f"- CSV: `{os.path.basename(csv_path)}`\n")
        handle.write(f"- Figure: `{os.path.basename(fig_path)}`\n")

    return csv_path, fig_path, report_path


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(
        description="Automatic Rayleigh SE validation for Cholesky/LDL/Jacobi with formula-CSV replay only"
    )
    parser.add_argument("--nr", type=int, default=64)
    parser.add_argument("--nt", type=int, default=16)
    parser.add_argument("--n-sc", type=int, default=168)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--snr-db", type=str, default="0,5,10,15,20")
    parser.add_argument("--pilot-len", type=int, default=16)
    parser.add_argument("--pilot-snr-db", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bj-layers", type=int, default=16)
    parser.add_argument(
        "--formula-csv-cholesky-noblock",
        type=str,
        default="result_new/cholesky/noblock/detailed_cycles_v3.csv",
    )
    parser.add_argument(
        "--formula-csv-cholesky-block",
        type=str,
        default="result_new/cholesky/block/detailed_cycles_v3.csv",
    )
    parser.add_argument(
        "--formula-csv-ldl-noblock",
        type=str,
        default="result_new/ldl/noblock/detailed_cycles_v3.csv",
    )
    parser.add_argument(
        "--formula-csv-ldl-block",
        type=str,
        default="result_new/ldl/block/detailed_cycles_v3.csv",
    )
    parser.add_argument(
        "--formula-csv-jacobi-noblock",
        type=str,
        default="",
    )
    parser.add_argument(
        "--formula-csv-jacobi-block",
        type=str,
        default="result_new/block_jacobi/operator/block_jacobi_cycle_detail.csv",
    )
    parser.add_argument("--out-dir", type=str, default="results/SE/three_algorithms_block_noblock_rayleigh")

    args = parser.parse_args()
    snr_list = [float(token.strip()) for token in args.snr_db.split(",") if token.strip()]

    return EvalConfig(
        nr=args.nr,
        nt=args.nt,
        n_sc=args.n_sc,
        batch=args.batch,
        trials=args.trials,
        snr_db_list=snr_list,
        pilot_len=args.pilot_len,
        pilot_snr_db=args.pilot_snr_db,
        seed=args.seed,
        bj_layers=args.bj_layers,
        formula_csv_cholesky_noblock=args.formula_csv_cholesky_noblock,
        formula_csv_cholesky_block=args.formula_csv_cholesky_block,
        formula_csv_ldl_noblock=args.formula_csv_ldl_noblock,
        formula_csv_ldl_block=args.formula_csv_ldl_block,
        formula_csv_jacobi_noblock=args.formula_csv_jacobi_noblock,
        formula_csv_jacobi_block=args.formula_csv_jacobi_block,
        out_dir=args.out_dir,
    )


def main() -> None:
    cfg = parse_args()
    if cfg.pilot_len < cfg.nt:
        raise SystemExit(f"pilot_len ({cfg.pilot_len}) must be >= nt ({cfg.nt})")
    if cfg.nt % 2 != 0:
        raise SystemExit(f"nt ({cfg.nt}) must be divisible by 2 for block variants")

    metrics = run_eval(cfg)
    outputs = save_outputs(cfg, metrics)

    print("SE validation finished.")
    print("Generated files:")
    for path in outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()
