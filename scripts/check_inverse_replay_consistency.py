#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.validate_se_block_noblock_rayleigh import (
    _load_formula_list,
    build_block_richardson_preconditioner,
    block_cholesky_inverse,
    block_ldl_inverse,
    replay_cholesky_inverse_from_formula_csv,
    replay_jacobi_inverse_from_formula_csv,
    replay_ldl_inverse_from_formula_csv,
)


@dataclass
class CheckCase:
    name: str
    csv_path: str
    solver: str
    block_size: int


def infer_jacobi_layers(formulas: list[str], default_layers: int) -> int:
    max_layer = 0
    patterns = [
        re.compile(r"^BY_\{(\d+)\}=B@Y_\{(\d+)\}$"),
        re.compile(r"^R_\{(\d+)\}=I-BY_\{(\d+)\}$"),
        re.compile(r"^Y_\{(\d+)\}=Y_\{(\d+)\}\+R_\{(\d+)\}$"),
    ]

    for formula in formulas:
        text = formula.replace(" ", "")
        for pattern in patterns:
            match = pattern.fullmatch(text)
            if match is None:
                continue
            for token in match.groups():
                max_layer = max(max_layer, int(token))

    return max(max_layer, default_layers)


def jacobi_formula_reference_inverse(a_mat: np.ndarray, block_size: int, n_layers: int) -> np.ndarray:
    b_mat, m_half_inv = build_block_richardson_preconditioner(a_mat, block_size=block_size)
    identity = np.eye(a_mat.shape[0], dtype=np.complex128)
    y_mat = identity.copy()
    for _ in range(n_layers):
        y_mat = y_mat + (identity - b_mat @ y_mat)
    return m_half_inv @ y_mat @ m_half_inv


def make_spd(nt: int, nr: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h_mat = (rng.normal(size=(nr, nt)) + 1j * rng.normal(size=(nr, nt))) / np.sqrt(2.0)
    return h_mat.conj().T @ h_mat + 1e-3 * np.eye(nt, dtype=np.complex128)


def run_case(case: CheckCase, nt: int, nr: int, seed: int, jacobi_layers: int) -> tuple[float, float]:
    formulas = _load_formula_list(case.csv_path)
    a_mat = make_spd(nt=nt, nr=nr, seed=seed)

    if case.solver == "cholesky":
        inv_ref = block_cholesky_inverse(a_mat, case.block_size)
        inv_replay = replay_cholesky_inverse_from_formula_csv(a_mat, case.block_size, formulas)
    elif case.solver == "ldl":
        inv_ref = block_ldl_inverse(a_mat, case.block_size)
        inv_replay = replay_ldl_inverse_from_formula_csv(a_mat, case.block_size, formulas)
    elif case.solver == "jacobi":
        n_layers = infer_jacobi_layers(formulas, jacobi_layers)
        inv_ref = jacobi_formula_reference_inverse(a_mat, case.block_size, n_layers=n_layers)
        inv_replay = replay_jacobi_inverse_from_formula_csv(a_mat, case.block_size, formulas, n_layers=n_layers)
    else:
        raise ValueError(f"Unknown solver: {case.solver}")

    rel_err = float(np.linalg.norm(inv_replay - inv_ref) / np.linalg.norm(inv_ref))
    resid = float(np.linalg.norm(a_mat @ inv_replay - np.eye(nt)) / np.linalg.norm(np.eye(nt)))
    return rel_err, resid


def main() -> int:
    parser = argparse.ArgumentParser(description="Check formula replay inverse consistency.")
    parser.add_argument("--nt", type=int, default=16)
    parser.add_argument("--nr", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--jacobi-layers", type=int, default=5)
    args = parser.parse_args()

    cases = [
        CheckCase("cholesky_noblock", "result_new/cholesky/noblock/detailed_cycles_v3.csv", "cholesky", 1),
        CheckCase("cholesky_block", "result_new/cholesky/block/detailed_cycles_v3.csv", "cholesky", 2),
        CheckCase("ldl_noblock", "result_new/ldl/noblock/detailed_cycles_v3.csv", "ldl", 1),
        CheckCase("ldl_block", "result_new/ldl/block/detailed_cycles_v3.csv", "ldl", 2),
        CheckCase("jacobi_block", "result_new/block_jacobi/operator/block_jacobi_cycle_detail.csv", "jacobi", 2),
    ]

    failed = False
    for case in cases:
        if not os.path.isfile(case.csv_path):
            print(f"[skip] {case.name}: missing {case.csv_path}")
            continue
        rel_err, resid = run_case(case, nt=args.nt, nr=args.nr, seed=args.seed, jacobi_layers=args.jacobi_layers)
        ok = rel_err <= args.tol
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {case.name}: rel_err={rel_err:.3e}, resid={resid:.3e}")
        failed = failed or (not ok)

    if failed:
        print("Consistency check failed.")
        return 2

    print("All replay consistency checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
