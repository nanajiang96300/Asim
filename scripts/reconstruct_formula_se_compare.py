#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.evaluate_ldl_quality import (  # noqa: E402
    EvalConfig,
    bits_to_16qam,
    bits_to_64qam,
    demod_16qam,
    demod_64qam,
    estimate_se,
    ldl_inverse,
    ls_channel_estimate,
    quantize_complex,
)


BRI_EIG_Q_LOW = 0.10
BRI_EIG_Q_HIGH = 0.90
BRI_EIG_MARGIN = 0.08
BRI_OMEGA_MIN = 0.0
BRI_OMEGA_MAX = 1.0e9
BRI_OMEGA_DAMP = 1.0


def finite_quantize(values: np.ndarray, eval_cfg: EvalConfig, clip: float = 6.0e4) -> np.ndarray:
    src = np.asarray(values, dtype=np.complex128)
    src = np.clip(src.real, -clip, clip) + 1j * np.clip(src.imag, -clip, clip)
    arr = quantize_complex(src, eval_cfg)
    arr = np.nan_to_num(arr, nan=0.0, posinf=clip, neginf=-clip)
    arr = np.clip(arr.real, -clip, clip) + 1j * np.clip(arr.imag, -clip, clip)
    return arr.astype(np.complex128, copy=False)


@dataclass
class CompareConfig:
    nr: int
    nt: int
    n_sc: int
    batch: int
    trials: int
    snr_db_list: List[float]
    pilot_len: int
    pilot_snr_db: float | None
    seed: int
    out_dir: str
    trunc_mantissa_bits: int
    modulation: str = "64qam"
    bj_dump_internals: bool = False
    bj_dump_dir: str = ""
    bj_dump_max_samples: int = 0


@dataclass
class FormulaModelSpec:
    name: str
    csv_path: Path


@dataclass
class FormulaModelMeta:
    name: str
    rows: List[dict]
    by_event: Dict[str, dict]
    inferred_layers: int
    inferred_block_size: int
    adaptive_bounds: bool
    use_iter_weight: bool


def read_formula_rows(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty formula csv: {path}")
    required = {"event_key", "formula", "compute_op", "step_idx"}
    missing = [key for key in required if key not in rows[0]]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    rows = sorted(rows, key=lambda r: int(r["step_idx"]))
    return rows


def infer_meta(name: str, rows: List[dict]) -> FormulaModelMeta:
    by_event = {r["event_key"]: r for r in rows}
    inferred_layers = 16
    inferred_block_size = 2
    adaptive_bounds = any(r["event_key"] == "ADAPTIVE_BOUNDS" for r in rows)
    use_iter_weight = any(re.match(r"^OMEGA_MUL_(\d+)$", r["event_key"]) for r in rows)

    if name == "block_richardson":
        layer_indices = []
        for event_key in by_event:
            match = re.match(r"^BY_(\d+)$", event_key)
            if match:
                layer_indices.append(int(match.group(1)))
        if layer_indices:
            inferred_layers = max(layer_indices) + 1

        block_hits: List[int] = []
        for event_key in by_event:
            match = re.search(r"B(\d+)_", event_key)
            if match:
                block_hits.append(int(match.group(1)))
        inferred_block_size = max(block_hits) if block_hits else 2

    if name == "ldl_noblock":
        inferred_block_size = 1
    if name == "ldl_block":
        inferred_block_size = 2

    return FormulaModelMeta(
        name=name,
        rows=rows,
        by_event=by_event,
        inferred_layers=inferred_layers,
        inferred_block_size=inferred_block_size,
        adaptive_bounds=adaptive_bounds,
        use_iter_weight=use_iter_weight,
    )


def validate_minimum_formula_coverage(meta: FormulaModelMeta) -> None:
    keyset = set(meta.by_event.keys())

    def has_prefix(prefix: str) -> bool:
        return any(key.startswith(prefix) for key in keyset)

    if meta.name in {"cholesky_block", "cholesky_noblock"}:
        required_exact = ["GRAM", "POTRF_DIAG_SQRT_0"]
        missing = [key for key in required_exact if key not in keyset]
        if missing:
            raise ValueError(f"{meta.name}: formula coverage missing keys {missing}")
        if not (has_prefix("TRSM_DIV_") or has_prefix("TRSM_MUL_")):
            raise ValueError(f"{meta.name}: missing TRSM formula steps")
        return

    if meta.name in {"ldl_block", "ldl_noblock"}:
        required_exact = ["GRAM", "REG", "D_UPDATE_0", "D_DIAG_INV_0", "D_INV_MUL_0"]
        missing = [key for key in required_exact if key not in keyset]
        if missing:
            raise ValueError(f"{meta.name}: formula coverage missing keys {missing}")
        return

    if meta.name == "block_richardson":
        required_exact = ["GRAM", "W", "XHAT", "BY_0", "RESIDUAL_0", "Y_UPDATE_0"]
        missing = [key for key in required_exact if key not in keyset]
        if missing:
            raise ValueError(f"{meta.name}: formula coverage missing keys {missing}")
        return


def generate_channel(rng: np.random.Generator, nr: int, nt: int) -> np.ndarray:
    return (rng.standard_normal((nr, nt)) + 1j * rng.standard_normal((nr, nt))) / np.sqrt(2.0)


def cholesky_formula_inverse(a_mat: np.ndarray, eval_cfg: EvalConfig) -> np.ndarray:
    n = a_mat.shape[0]
    a = quantize_complex(a_mat, eval_cfg)
    l_mat = np.zeros_like(a, dtype=np.complex128)

    for j in range(n):
        if j > 0:
            diag_acc = np.dot(l_mat[j, :j], l_mat[j, :j].conj())
        else:
            diag_acc = 0.0
        diag_term = quantize_complex(np.array([a[j, j] - diag_acc]), eval_cfg)[0]
        l_jj = quantize_complex(np.array([np.sqrt(diag_term)]), eval_cfg)[0]
        l_mat[j, j] = l_jj

        inv_l_jj = quantize_complex(np.array([1.0 / l_jj]), eval_cfg)[0]

        for i in range(j + 1, n):
            if j > 0:
                off_acc = np.dot(l_mat[i, :j], l_mat[j, :j].conj())
            else:
                off_acc = 0.0
            num = quantize_complex(np.array([a[i, j] - off_acc]), eval_cfg)[0]
            l_mat[i, j] = quantize_complex(np.array([num * inv_l_jj]), eval_cfg)[0]

    identity = np.eye(n, dtype=np.complex128)
    y = np.linalg.solve(l_mat, identity)
    y = quantize_complex(y, eval_cfg)
    a_inv = quantize_complex(y.conj().T @ y, eval_cfg)
    return a_inv


def invert_spd_block(block: np.ndarray, solver: str) -> np.ndarray:
    if solver == "direct2x2":
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

    l_mat = np.linalg.cholesky(block)
    eye = np.eye(block.shape[0], dtype=np.complex128)
    y_mat = np.linalg.solve(l_mat, eye)
    return np.linalg.solve(l_mat.conj().T, y_mat)


def build_block_richardson_preconditioner(a_mat: np.ndarray, blk: int, precond_solver: str) -> tuple[np.ndarray, np.ndarray]:
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


def chebyshev_omega(n_layers: int, bmin: float = 0.1, bmax: float = 1.2) -> List[float]:
    omegas = []
    bmin = max(float(bmin), 1e-8)
    bmax = max(float(bmax), bmin + 1e-8)
    for layer in range(n_layers):
        theta = np.pi * (2 * layer + 1) / (2 * n_layers)
        dt = 0.5 * (bmax + bmin) + 0.5 * (bmax - bmin) * np.cos(theta)
        omegas.append(float(1.0 / dt))
    return omegas


def chebyshev_omega_adaptive(b_mat: np.ndarray, n_layers: int, nt: int, floor: float = 1e-8) -> List[float]:
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


def block_richardson_formula_inverse(
    a_mat: np.ndarray,
    meta: FormulaModelMeta,
    eval_cfg: EvalConfig,
    debug_collector: Dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    precond_solver = "direct2x2" if meta.inferred_block_size == 2 else "cholesky"
    b_mat, b_inv = build_block_richardson_preconditioner(a_mat, blk=meta.inferred_block_size, precond_solver=precond_solver)
    y_mat = np.zeros_like(a_mat, dtype=np.complex128)
    identity = np.eye(a_mat.shape[0], dtype=np.complex128)

    if not meta.use_iter_weight:
        omegas = [1.0] * max(meta.inferred_layers, 1)
    elif meta.adaptive_bounds:
        omegas = chebyshev_omega_adaptive(b_mat, meta.inferred_layers, nt=int(a_mat.shape[0]))
    else:
        omegas = chebyshev_omega(meta.inferred_layers, bmin=0.1, bmax=1.2)

    residual_norms: List[float] = []
    y_snapshots: List[np.ndarray] = []
    for omega in omegas:
        residual = identity - b_mat @ y_mat
        res_norm = float(np.linalg.norm(residual))
        residual_norms.append(res_norm)
        y_mat = y_mat + omega * residual
        if debug_collector is not None:
            y_snapshots.append(np.asarray(y_mat, dtype=np.complex128))

    a_inv = y_mat @ b_inv
    if debug_collector is not None:
        debug_collector["a_est"] = np.asarray(a_mat, dtype=np.complex128)
        debug_collector["b_mat"] = np.asarray(b_mat, dtype=np.complex128)
        debug_collector["b_inv"] = np.asarray(b_inv, dtype=np.complex128)
        debug_collector["omegas"] = np.asarray(omegas, dtype=np.float64)
        debug_collector["residual_norms"] = np.asarray(residual_norms, dtype=np.float64)
        debug_collector["y_last"] = np.asarray(y_mat, dtype=np.complex128)
        if y_snapshots:
            debug_collector["y_layers"] = np.stack(y_snapshots, axis=0)
    return a_inv


def ldl_noblock_formula_inverse(a_mat: np.ndarray, eval_cfg: EvalConfig) -> np.ndarray:
    n = a_mat.shape[0]
    a = quantize_complex(a_mat, eval_cfg)
    l_mat = np.eye(n, dtype=np.complex128)
    d_vec = np.zeros(n, dtype=np.complex128)

    for j in range(n):
        diag_acc = 0.0 + 0.0j
        for k in range(j):
            diag_acc += l_mat[j, k] * d_vec[k] * np.conj(l_mat[j, k])
        d_vec[j] = quantize_complex(np.array([a[j, j] - diag_acc]), eval_cfg)[0]

        inv_dj = quantize_complex(np.array([1.0 / (d_vec[j] + 1e-12)]), eval_cfg)[0]
        for i in range(j + 1, n):
            off_acc = 0.0 + 0.0j
            for k in range(j):
                off_acc += l_mat[i, k] * d_vec[k] * np.conj(l_mat[j, k])
            num = quantize_complex(np.array([a[i, j] - off_acc]), eval_cfg)[0]
            l_mat[i, j] = quantize_complex(np.array([num * inv_dj]), eval_cfg)[0]

    inv_l = np.linalg.inv(l_mat)
    inv_d = np.diag(quantize_complex(1.0 / (d_vec + 1e-12), eval_cfg))
    a_inv = quantize_complex(inv_l.conj().T @ inv_d @ inv_l, eval_cfg)
    return a_inv


def inverse_from_formula(
    meta: FormulaModelMeta,
    a_est: np.ndarray,
    eval_cfg: EvalConfig,
    debug_collector: Dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    if meta.name in {"cholesky_block", "cholesky_noblock"}:
        return cholesky_formula_inverse(a_est, eval_cfg)
    if meta.name == "ldl_noblock":
        return ldl_noblock_formula_inverse(a_est, eval_cfg)
    if meta.name == "ldl_block":
        block_size = 2
        a_inv, _ = ldl_inverse(a_est, eval_cfg, block_size=block_size)
        return a_inv
    if meta.name == "block_richardson":
        return block_richardson_formula_inverse(a_est, meta, eval_cfg, debug_collector=debug_collector)
    raise ValueError(f"Unknown model: {meta.name}")


def run_eval(cfg: CompareConfig, specs: List[FormulaModelSpec]) -> List[dict]:
    rng = np.random.default_rng(cfg.seed)
    metas = [infer_meta(spec.name, read_formula_rows(spec.csv_path)) for spec in specs]
    for meta in metas:
        validate_minimum_formula_coverage(meta)

    eval_cfg = EvalConfig(
        nr=cfg.nr,
        nt=cfg.nt,
        n_sc=cfg.n_sc,
        batch=cfg.batch,
        trials=cfg.trials,
        block_size=2,
        snr_db_list=cfg.snr_db_list,
        channel_model="rayleigh",
        pilot_len=cfg.pilot_len,
        pilot_snr_db=cfg.pilot_snr_db,
        num_format="fp16",
        reciprocal_mode="exact",
        trunc_mantissa_bits=cfg.trunc_mantissa_bits,
        modulation=cfg.modulation,
        mac_chunk=4,
        seed=cfg.seed,
        out_dir=cfg.out_dir,
    )

    rows_out = []
    dump_manifest_rows: List[dict] = []
    dump_count = 0
    dump_dir = Path(cfg.bj_dump_dir) if cfg.bj_dump_dir else (Path(cfg.out_dir) / "bj_internal_dump")
    if cfg.bj_dump_internals:
        dump_dir.mkdir(parents=True, exist_ok=True)
    for snr_db in cfg.snr_db_list:
        snr_lin = 10.0 ** (snr_db / 10.0)
        noise_var = 1.0 / snr_lin
        pilot_snr_lin = snr_lin if cfg.pilot_snr_db is None else 10.0 ** (cfg.pilot_snr_db / 10.0)
        pilot_noise_var = 1.0 / pilot_snr_lin

        se_sum = {meta.name: 0.0 for meta in metas}
        total_samples = cfg.trials * cfg.batch * cfg.n_sc

        for sample_idx in range(total_samples):
            h_true = generate_channel(rng, cfg.nr, cfg.nt)
            h_est = ls_channel_estimate(rng, h_true, cfg.pilot_len, pilot_noise_var)

            if cfg.modulation == "16qam":
                tx_bits = rng.integers(0, 2, size=(cfg.nt, 4), dtype=np.int32)
                x_tx = bits_to_16qam(tx_bits)
            elif cfg.modulation == "64qam":
                tx_bits = rng.integers(0, 2, size=(cfg.nt, 6), dtype=np.int32)
                x_tx = bits_to_64qam(tx_bits)
            else:
                tx_bits = rng.integers(0, 2, size=(cfg.nt, 1), dtype=np.int32)
                x_tx = (2 * tx_bits[:, 0] - 1).astype(np.float64).astype(np.complex128)

            noise = np.sqrt(noise_var / 2.0) * (
                rng.standard_normal(cfg.nr) + 1j * rng.standard_normal(cfg.nr)
            )
            y_rx = h_true @ x_tx + noise

            a_est = h_est.conj().T @ h_est + noise_var * np.eye(cfg.nt, dtype=np.complex128)
            h_h_est = h_est.conj().T

            for meta in metas:
                debug_payload: Dict[str, np.ndarray] | None = None
                should_dump = (
                    cfg.bj_dump_internals
                    and meta.name == "block_richardson"
                    and dump_count < cfg.bj_dump_max_samples
                )
                if should_dump:
                    debug_payload = {}

                a_inv = inverse_from_formula(meta, a_est, eval_cfg, debug_collector=debug_payload)
                w = quantize_complex(a_inv @ h_h_est, eval_cfg)
                x_hat = quantize_complex(w @ y_rx, eval_cfg)
                if cfg.modulation == "16qam":
                    _ = demod_16qam(x_hat)
                elif cfg.modulation == "64qam":
                    _ = demod_64qam(x_hat)
                else:
                    _ = (np.real(x_hat) >= 0).astype(np.int32)
                se_sum[meta.name] += estimate_se(w, h_true, noise_var, cfg.modulation)

                if should_dump and debug_payload is not None:
                    dump_file = dump_dir / f"bj_internal_snr{snr_db:g}_sample{sample_idx:04d}.npz"
                    np.savez_compressed(dump_file, **debug_payload)
                    dump_manifest_rows.append(
                        {
                            "snr_db": float(snr_db),
                            "sample_idx": sample_idx,
                            "file": str(dump_file),
                            "layers": int(meta.inferred_layers),
                            "block_size": int(meta.inferred_block_size),
                            "adaptive_bounds": int(meta.adaptive_bounds),
                            "use_iter_weight": int(meta.use_iter_weight),
                        }
                    )
                    dump_count += 1

        row = {"snr_db": float(snr_db)}
        for meta in metas:
            row[f"se_{meta.name}"] = se_sum[meta.name] / max(total_samples, 1)
        rows_out.append(row)
        print(f"[done] snr={snr_db} dB -> " + ", ".join([f"{k}={row[k]:.4f}" for k in row if k != "snr_db"]))

    if cfg.bj_dump_internals and dump_manifest_rows:
        manifest_path = dump_dir / "manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dump_manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(dump_manifest_rows)
        print(f"saved_bj_internal_manifest={manifest_path}")

    return rows_out


def save_outputs(cfg: CompareConfig, rows: List[dict], specs: List[FormulaModelSpec]) -> tuple[Path, Path]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "se_formula_reconstruct_5alg_rayleigh_ls_fp16.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    png_path = out_dir / "se_formula_reconstruct_5alg_rayleigh_ls_fp16.png"
    plt.figure(figsize=(7.8, 5.2))

    label_map = {
        "cholesky_block": "Cholesky Block (formula)",
        "cholesky_noblock": "Cholesky NoBlock (formula)",
        "ldl_block": "LDL Block (formula)",
        "ldl_noblock": "LDL NoBlock (formula)",
        "block_richardson": "Block Jacobi (formula)",
    }

    snr = [r["snr_db"] for r in rows]
    for spec in specs:
        key = f"se_{spec.name}"
        plt.plot(snr, [r[key] for r in rows], marker="o", label=label_map.get(spec.name, spec.name))

    plt.xlabel("SNR (dB)")
    plt.ylabel("SE (bit/s/Hz)")
    plt.title("Formula-Reconstructed SE Comparison (Rayleigh + LS + FP16)")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)

    return csv_path, png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct 5 algorithms from formula cycle tables and compare SE"
    )
    parser.add_argument("--nr", type=int, default=64)
    parser.add_argument("--nt", type=int, default=16)
    parser.add_argument("--n-sc", type=int, default=16)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--snr-db", type=str, default="0,5,10,15,20,25,30")
    parser.add_argument("--pilot-len", type=int, default=16)
    parser.add_argument("--pilot-snr-db", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modulation", type=str, default="64qam", choices=["16qam", "64qam", "bpsk"])
    parser.add_argument("--trunc-mantissa-bits", type=int, default=10)
    parser.add_argument("--bj-dump-internals", action="store_true")
    parser.add_argument("--bj-dump-dir", type=str, default="")
    parser.add_argument("--bj-dump-max-samples", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="result_new/formula_reconstruct_se_compare",
    )

    parser.add_argument("--chol-block-csv", type=str, default="result_new/cholesky/block/detailed_cycles_v3.csv")
    parser.add_argument("--chol-noblock-csv", type=str, default="result_new/cholesky/noblock/detailed_cycles_v3.csv")
    parser.add_argument("--ldl-block-csv", type=str, default="result_new/ldl/block/detailed_cycles_v3.csv")
    parser.add_argument("--ldl-noblock-csv", type=str, default="result_new/ldl/noblock/detailed_cycles_v3.csv")
    parser.add_argument(
        "--block-jacobi-csv",
        type=str,
        default="result_new/block_richardson/operator/block_richardson_cycle_detail.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = CompareConfig(
        nr=args.nr,
        nt=args.nt,
        n_sc=args.n_sc,
        batch=args.batch,
        trials=args.trials,
        snr_db_list=[float(x.strip()) for x in args.snr_db.split(",") if x.strip()],
        pilot_len=args.pilot_len,
        pilot_snr_db=args.pilot_snr_db,
        seed=args.seed,
        out_dir=args.out_dir,
        trunc_mantissa_bits=args.trunc_mantissa_bits,
        modulation=args.modulation,
        bj_dump_internals=args.bj_dump_internals,
        bj_dump_dir=args.bj_dump_dir,
        bj_dump_max_samples=args.bj_dump_max_samples,
    )

    specs = [
        FormulaModelSpec("cholesky_block", Path(args.chol_block_csv)),
        FormulaModelSpec("cholesky_noblock", Path(args.chol_noblock_csv)),
        FormulaModelSpec("ldl_block", Path(args.ldl_block_csv)),
        FormulaModelSpec("ldl_noblock", Path(args.ldl_noblock_csv)),
        FormulaModelSpec("block_richardson", Path(args.block_richardson_csv)),
    ]

    for spec in specs:
        if not spec.csv_path.exists():
            raise FileNotFoundError(f"missing formula csv: {spec.csv_path}")

    rows = run_eval(cfg, specs)
    csv_path, png_path = save_outputs(cfg, rows, specs)

    print(f"saved_csv={csv_path}")
    print(f"saved_png={png_path}")


if __name__ == "__main__":
    main()
