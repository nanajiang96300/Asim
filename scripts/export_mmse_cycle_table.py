#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path
from typing import Dict, List, Tuple


def normalize_compute_op_by_units(compute_op: str, units: set[str]) -> str:
    if not units:
        return compute_op
    all_scalar = all("Scalar" in unit for unit in units)
    all_vector = all("Vector" in unit for unit in units)
    if compute_op.startswith("VECTOR_") and all_scalar:
        return "SCALAR_" + compute_op[len("VECTOR_"):]
    if compute_op.startswith("SCALAR_") and all_vector:
        return "VECTOR_" + compute_op[len("SCALAR_"):]
    return compute_op


def map_mmse_event(event_key: str, m: int, u: int) -> Tuple[int, str, str, str, str, str]:
    if event_key == "HtH":
        return -1, "GRAM", "MatMul", "MATMUL", "G = H^H @ H", f"{u}*{m}*{m}*{u}"
    if event_key == "G_PLUS_SIGMA":
        return -1, "REG", "Add", "VECTOR_ADD", "A = G + RegI", f"{u}*{u}"
    if event_key == "WH":
        return -1, "W", "MatMul", "MATMUL", "W = A^{-1} @ H^H", f"{u}*{u}*{u}*{m}"
    if event_key == "WY":
        return -1, "XHAT", "MatMul", "MATMUL", "X_hat = W @ Y", f"{u}*{m}*{m}*{u}"

    mobj = re.match(r"^INV_T$", event_key)
    if mobj:
        return -1, "NS_INV_T", "MatMul", "MATMUL", "T = A @ X", f"{u}*{u}*{u}*{u}"
    mobj = re.match(r"^INV_R$", event_key)
    if mobj:
        return -1, "NS_INV_R", "Add", "VECTOR_ADD", "R = C - T", f"{u}*{u}"
    mobj = re.match(r"^INV_X$", event_key)
    if mobj:
        return -1, "NS_INV_X", "MatMul", "MATMUL", "X = X @ R", f"{u}*{u}*{u}*{u}"

    mobj = re.match(r"^CHOL(_ISO)?_POTRF_DIAG_UPD_(\d+)(?:_(\d+))?$", event_key)
    if mobj:
        j = int(mobj.group(2))
        return j, f"CHOL_POTRF_DIAG_UPD_{j}", "SubMul", "VECTOR_MAC", "A_diag = A_diag - L*L", "2"
    mobj = re.match(r"^CHOL(_ISO)?_POTRF_DIAG_SQRT_(\d+)$", event_key)
    if mobj:
        j = int(mobj.group(2))
        return j, f"CHOL_POTRF_DIAG_SQRT_{j}", "Sqrt", "SCALAR_SQRT", "L_diag = sqrt(A_diag)", "2*2"
    mobj = re.match(r"^CHOL(_ISO)?_TRSM_NUM_UPD_(\d+)_(\d+)(?:_(\d+))?$", event_key)
    if mobj:
        i = int(mobj.group(2))
        j = int(mobj.group(3))
        return i, f"CHOL_TRSM_NUM_UPD_{i}_{j}", "SubMul", "VECTOR_MAC", "A_off = A_off - L*L", "2"
    mobj = re.match(r"^CHOL(_ISO)?_TRSM_DIV_(\d+)_(\d+)$", event_key)
    if mobj:
        i = int(mobj.group(2))
        j = int(mobj.group(3))
        return i, f"CHOL_TRSM_DIV_{i}_{j}", "Div", "SCALAR_DIV", "L_off = A_off / L_diag", "2*2"
    mobj = re.match(r"^CHOL(_ISO)?_RK_UPDATE_(\d+)(?:_(\d+))?(?:_(\d+))?$", event_key)
    if mobj:
        i = int(mobj.group(2))
        return i, f"CHOL_RK_UPDATE_{i}", "SubMul", "VECTOR_MAC", "A = A - L*L^T", "2"
    mobj = re.match(r"^CHOL(_ISO)?_FWD_DIAG_INV_(\d+)$", event_key)
    if mobj:
        c = int(mobj.group(2))
        return c, f"CHOL_FWD_DIAG_INV_{c}", "Div", "SCALAR_DIV", "Y_diag = 1/L_diag", "2*2"
    mobj = re.match(r"^CHOL(_ISO)?_FWD_OFF_MAC_(\d+)_(\d+)_(\d+)$", event_key)
    if mobj:
        i = int(mobj.group(2))
        c = int(mobj.group(3))
        rep = int(mobj.group(4))
        return i, f"CHOL_FWD_OFF_MAC_{i}_{c}_{rep}", "SubMul", "VECTOR_MAC", "tmp = Y - L*Y", "2"
    mobj = re.match(r"^CHOL(_ISO)?_FWD_OFF_UPD_(\d+)_(\d+)_(\d+)$", event_key)
    if mobj:
        i = int(mobj.group(2))
        c = int(mobj.group(3))
        rep = int(mobj.group(4))
        return i, f"CHOL_FWD_OFF_UPD_{i}_{c}_{rep}", "Div", "SCALAR_DIV", "Y_off = tmp/L_diag", "2*2"
    if event_key in {"CHOL_BWD_MAC_FULL", "CHOL_ISO_BWD_MAC_FULL"}:
        return -1, "CHOL_BWD_MAC_FULL", "MatMul", "MATMUL", "A_inv = Y^T @ Y", f"{u}*{u}*{u}*{u}"

    return -1, event_key, "Unknown", "VECTOR_ADD", event_key, ""


def keep_mmse_repro_event(event_key: str) -> bool:
    if event_key in {"GRAM", "REG", "W", "XHAT"}:
        return True
    if re.match(r"^CHOL_POTRF_DIAG_SQRT_\d+$", event_key):
        return True
    if re.match(r"^CHOL_TRSM_DIV_\d+_\d+$", event_key):
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MMSE operator cycle detail table")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrix-m", type=int, default=64)
    parser.add_argument("--matrix-u", type=int, default=16)
    parser.add_argument("--reducer", type=str, default="median", choices=["median", "mean", "max", "sum"])
    args = parser.parse_args()

    grouped: Dict[str, List[dict]] = {}
    with args.trace.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row["name"]
            if name in {"Load", "Store", "CubeWait"}:
                continue
            if not name.startswith("MMSE_"):
                continue
            event_key = name[5:]
            if event_key.startswith("BARRIER_") or event_key.startswith("OUT"):
                continue
            grouped.setdefault(name, []).append(row)

    rows: List[dict] = []
    major_buckets: Dict[Tuple[str, int], List[float]] = {}
    major_substeps: Dict[Tuple[str, int], set] = {}
    major_ops: Dict[Tuple[str, int], set] = {}

    for idx, full_name in enumerate(sorted(grouped.keys())):
        events = grouped[full_name]
        event_key = full_name[5:] if full_name.startswith("MMSE_") else full_name
        layer_idx, major_step, onnx_op, comp, formula, dims = map_mmse_event(event_key, args.matrix_m, args.matrix_u)
        if not keep_mmse_repro_event(major_step):
            continue

        units = sorted({e["unit"] for e in events})
        durations = [int(e["end_cycle"]) - int(e["start_cycle"]) for e in events]
        durations = [d for d in durations if d >= 0]
        if args.reducer == "median":
            cycle = statistics.median(durations) if durations else 0.0
        elif args.reducer == "mean":
            cycle = statistics.mean(durations) if durations else 0.0
        elif args.reducer == "max":
            cycle = max(durations) if durations else 0.0
        else:
            cycle = sum(durations) if durations else 0.0
        comp_norm = normalize_compute_op_by_units(comp, set(units))

        rows.append(
            {
                "step_idx": str(idx),
                "layer_idx": str(layer_idx),
                "operator_mode": "mmse_cholesky_baseline",
                "major_step": major_step,
                "event_key": major_step,
                "onnx_op": onnx_op,
                "compute_op": comp_norm,
                "formula": formula,
                "formula_dims": dims,
                "compute_cycles": f"{cycle:.2f}",
                "matched_events": full_name,
                "matched_units": "|".join(units),
            }
        )

        mkey = (major_step, layer_idx)
        major_buckets.setdefault(mkey, []).append(cycle)
        major_substeps.setdefault(mkey, set()).add(event_key)
        major_ops.setdefault(mkey, set()).add(comp_norm)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step_idx",
                "layer_idx",
                "operator_mode",
                "major_step",
                "event_key",
                "onnx_op",
                "compute_op",
                "formula",
                "formula_dims",
                "compute_cycles",
                "matched_events",
                "matched_units",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    major_csv = args.output.with_name(args.output.stem + "_major_summary.csv")
    major_rows: List[dict] = []
    for idx, ((major_step, layer_idx), vals) in enumerate(sorted(major_buckets.items(), key=lambda kv: (kv[0][1], kv[0][0]))):
        major_rows.append(
            {
                "major_idx": str(idx),
                "layer_idx": str(layer_idx),
                "operator_mode": "mmse_cholesky_baseline",
                "major_step": major_step,
                "compute_ops": "|".join(sorted(major_ops[(major_step, layer_idx)])),
                "substep_count": str(len(major_substeps[(major_step, layer_idx)])),
                "major_cycle_sum": f"{sum(vals):.2f}",
                "major_cycle_mean": f"{statistics.mean(vals):.2f}",
                "major_cycle_max": f"{max(vals):.2f}",
                "major_cycle_min": f"{min(vals):.2f}",
                "substeps": "|".join(sorted(major_substeps[(major_step, layer_idx)])),
            }
        )

    with major_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "major_idx",
                "layer_idx",
                "operator_mode",
                "major_step",
                "compute_ops",
                "substep_count",
                "major_cycle_sum",
                "major_cycle_mean",
                "major_cycle_max",
                "major_cycle_min",
                "substeps",
            ],
        )
        writer.writeheader()
        writer.writerows(major_rows)

    print(f"mode=mmse_cholesky_baseline")
    print(f"detail_rows={len(rows)}")
    print(f"major_rows={len(major_rows)}")
    print(f"output={args.output}")
    print(f"summary_output={major_csv}")


if __name__ == "__main__":
    main()
