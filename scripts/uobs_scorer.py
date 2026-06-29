#!/usr/bin/env python3
"""
UOBS Black-Box Scorer (Phase 2) — Clean version.
Reads formula_steps.json + trace.csv, verifies numerical correctness using
existing validated formula-inverse implementations, then computes a single
scalar score.

Operator-agnostic: algorithm is identified from formula_steps.json signature.
Black-box to AI: only final score is exposed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.reference_inverse_registry import (  # noqa: E402
    compute_reference_inverse,
    list_algorithms,
)

# ── Algorithm identification ───────────────────────────────────────────────

@dataclass
class AlgorithmIdentity:
    name: str
    block_size: int = 2
    layers: int = 16
    has_adaptive_omega: bool = False
    total_steps: int = 0


def identify_algorithm(formula_json_path: str) -> AlgorithmIdentity:
    with open(formula_json_path, "r") as f:
        data = json.load(f)

    # New format: {"_metadata": {...}, "steps": [...]}
    # Old format: [{...}, {...}]  (plain array)
    if isinstance(data, dict):
        meta = data.get("_metadata", {})
        steps = data.get("steps", [])
        # If metadata declares algorithm explicitly, use it directly
        algo_name = meta.get("algorithm", "")
        if algo_name:
            return AlgorithmIdentity(
                name=algo_name,
                block_size=meta.get("block_size", 2) or 2,
                layers=meta.get("layers", 0) or 16,
                has_adaptive_omega=("adaptive" in algo_name),
                total_steps=len(steps))
    else:
        steps = data
        meta = {}

    if not steps:
        raise ValueError("Empty formula_steps.json")

    step_ids = [s["step_id"] for s in steps]
    # Heuristic: check step_id naming patterns
    has_chol = any("CHOL_BLOCK" in s or "CHOL_NB" in s for s in step_ids)
    has_ldl = any("LDL_BLOCK" in s for s in step_ids)
    has_bri = any("BRI_" in s for s in step_ids)
    has_ns = any(s.startswith("NSOPT_") for s in step_ids)
    has_potrf = any("POTRF" in s for s in step_ids)
    has_trsm = any("TRSM" in s for s in step_ids)
    has_d_update = any("D_UPDATE" in s for s in step_ids)
    has_l_update = any("L_UPDATE" in s for s in step_ids)
    has_bri_by = any(s.startswith("BRI_BY_") for s in step_ids)
    has_bri_res = any(s.startswith("BRI_RESIDUAL") for s in step_ids)

    block_size = 2
    has_noblock_prefix = any("CHOL_NB" in s or "LDL_NB" in s for s in step_ids)
    if has_noblock_prefix:
        block_size = 1
    for s in steps:
        if s["op_type"] == "CHOLESKY":
            bs = s["output_shape"][0]
            if bs > block_size:
                block_size = bs
    if not has_potrf and block_size == 2 and not has_noblock_prefix:
        # Detect no-block for LDL: check the first GEMM step's output dimension.
        # If output_shape matches the full matrix size (nt), and no sub-block
        # GEMM steps exist, it's likely a no-block variant.
        # Fallback: use the trace to detect block_size from config.
        pass  # Deferred to refine_from_trace

    if not has_potrf:
        for s in steps:
            if s["op_type"] == "DIAG_INV" and s["output_shape"][0] >= 2:
                block_size = s["output_shape"][0]
                break

    layers = 16
    if has_bri_by:
        indices = []
        for s in steps:
            sid = s["step_id"]
            if sid.startswith("BRI_BY_"):
                try:
                    indices.append(int(sid.split("_")[-1]))
                except ValueError:
                    pass
        if indices:
            layers = max(indices) + 1

    adaptive = any("OMEGA_MUL" in s for s in step_ids)

    # Extract NS iteration count
    ns_layers = 5
    if has_ns:
        indices = []
        for s in steps:
            sid = s["step_id"]
            if sid.startswith("NSOPT_GEMM_T_"):
                try:
                    indices.append(int(sid.split("_")[-1]))
                except ValueError:
                    pass
        if indices:
            ns_layers = max(indices) + 1

    if has_ns:
        name = "newton_schulz"
        layers = ns_layers
    elif has_bri_by and has_bri_res:
        name = "block_richardson"
    elif has_bri:
        name = "block_richardson"
    elif has_d_update or has_l_update or has_ldl:
        name = "ldl_block" if block_size > 1 else "ldl_noblock"
    elif has_potrf or has_trsm or has_chol:
        name = "cholesky_block" if block_size > 1 else "cholesky_noblock"
    else:
        raise ValueError(f"Cannot identify algorithm from steps: {step_ids[:10]}...")

    return AlgorithmIdentity(name=name, block_size=block_size, layers=layers,
                             has_adaptive_omega=adaptive, total_steps=len(steps))


def refine_from_trace(algo: AlgorithmIdentity, trace_csv_path: str) -> AlgorithmIdentity:
    """Refine algorithm parameters using trace.csv (e.g. actual layers for BJ)."""
    if algo.name != "block_richardson":
        return algo
    try:
        with open(trace_csv_path) as f:
            rows = list(csv.DictReader(f))
        layers_set = set()
        has_omega = False
        for r in rows:
            name = str(r.get("name", ""))
            if name.startswith("BRI_BY_"):
                parts = name.replace("BRI_BY_", "").split("_")
                for p in parts:
                    try:
                        layers_set.add(int(p))
                        break
                    except ValueError:
                        pass
            if "OMEGA_MUL" in name:
                has_omega = True
        if layers_set:
            algo.layers = max(layers_set) + 1
        if has_omega:
            algo.has_adaptive_omega = True
    except Exception:
        pass
    return algo



def compute_pipeline_parallelism(trace_csv_path: str, core_id: int = 0) -> float:
    """
    Compute average pipeline parallelism for a given core.
    Counts how many functional units (Cube, Vector, Scalar, MTE2, MTE3)
    are active per cycle, averaged across all cycles with activity.
    Returns the avg_parallelism score (higher = better pipeline overlap).
    """
    core_prefix = f"Core{core_id}_"
    events = []
    max_end = 0
    try:
        with open(trace_csv_path, "r") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return 0.0

    for r in rows:
        unit = str(r.get("unit", "")).strip('\"')
        if not unit.startswith(core_prefix):
            continue
        try:
            start = int(float(r["start_cycle"]))
            end = int(float(r["end_cycle"]))
            if start >= 0 and end > start:
                u = unit.split("_", 1)[1]
                if u in ("Cube", "Vector", "Scalar", "MTE2", "MTE3"):
                    events.append((u, start, end))
                    if end > max_end:
                        max_end = end
        except (ValueError, KeyError):
            continue

    if not events or max_end == 0:
        return 0.0

    span = min(max_end, 500000)  # cap for memory safety
    if max_end > 500000:
        return _sparse_parallelism(events, max_end)

    active_mask = [0] * span
    unit_to_bit = {"Cube": 1, "Vector": 2, "Scalar": 4, "MTE2": 8, "MTE3": 16}

    for u, start, end in events:
        if u in unit_to_bit:
            bit = unit_to_bit[u]
            for c in range(start, min(end, span)):
                active_mask[c] |= bit

    total_units = 0
    total_active = 0
    for mask in active_mask:
        count = bin(mask).count("1")
        if count > 0:
            total_units += count
            total_active += 1

    return round(total_units / max(total_active, 1), 4)


def _sparse_parallelism(events: list, max_end: int) -> float:
    """Fallback for long traces: compute parallelism from merged intervals."""
    from collections import defaultdict
    unit_intervals = defaultdict(list)
    for u, s, e in events:
        unit_intervals[u].append((s, e))

    merged = {}
    for u, intervals in unit_intervals.items():
        intervals.sort()
        merged[u] = []
        for s, e in intervals:
            if merged[u] and s <= merged[u][-1][1]:
                merged[u][-1] = (merged[u][-1][0], max(merged[u][-1][1], e))
            else:
                merged[u].append((s, e))

    points = []
    for u, intervals in merged.items():
        for s, e in intervals:
            points.append((s, 1))
            points.append((e, -1))
    points.sort()

    active = 0
    total_units = 0
    total_active = 0
    prev = 0
    for t, d in points:
        if t > prev and active > 0:
            length = t - prev
            total_units += active * length
            total_active += length
        active += d
        prev = t
    return round(total_units / max(total_active, 1), 4)


def extract_trace_stats(trace_csv_path: str) -> Dict[str, float]:
    with open(trace_csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"finish_cycle": 0, "cube_util": 0.0, "cube_events": 0}

    sc, ec, uc = "start_cycle", "end_cycle", "unit"

    cycles = []
    for r in rows:
        try:
            cs, ce = int(float(r[sc])), int(float(r[ec]))
            if cs >= 0 and ce > cs:
                cycles.append((cs, ce))
        except (ValueError, KeyError):
            continue
    if not cycles:
        return {"finish_cycle": 0, "cube_util": 0.0, "cube_events": 0}

    gs, ge = min(c[0] for c in cycles), max(c[1] for c in cycles)
    span = ge - gs

    # Per-core Cube utilisation (unit format: "CoreX_Cube")
    core_cube: Dict[str, float] = {}
    cube_ev = 0
    for r in rows:
        unit = str(r.get(uc, ""))
        if "_Cube" in unit:
            try:
                dur = int(float(r[ec])) - int(float(r[sc]))
                core = unit.split("_")[0]
                core_cube[core] = core_cube.get(core, 0) + dur
                cube_ev += 1
            except (ValueError, KeyError):
                pass
    avg_cu = (np.mean([d / max(span, 1) * 100 for d in core_cube.values()])
              if core_cube else 0.0)

    # Pipeline parallelism metric
    avg_par = compute_pipeline_parallelism(trace_csv_path, core_id=0)

    return {"finish_cycle": ge, "cube_util": round(avg_cu, 4),
            "cube_events": cube_ev, "total_span": span,
            "total_events": len(cycles),
            "avg_parallelism": avg_par}


# ── Formula-Trace Cross-Validation ──────────────────────────────────────────

def validate_trace_formula_consistency(formula_json_path: str,
                                       trace_csv_path: str) -> Tuple[bool, str]:
    """Cross-validate that trace actually executed the operations declared in formula.

    Returns (is_valid, reason). Catches the class of bugs where an agent generates
    correct FormulaLogger declarations but broken/incomplete instruction sequences,
    producing artificially low cycle counts that still pass SE validation.
    """
    with open(formula_json_path, "r") as f:
        data = json.load(f)
    steps = data.get("steps", []) if isinstance(data, dict) else data

    # Count GEMM-type formula steps
    formula_gemm = sum(1 for s in steps
                       if isinstance(s, dict) and s.get("op_type") == "GEMM")

    # Count Cube pipeline ops in trace
    trace_cube = 0
    try:
        with open(trace_csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                unit = str(row.get("unit", ""))
                if "_Cube" in unit:
                    trace_cube += 1
    except Exception:
        return True, ""  # Can't read trace, skip validation

    # If formula declares GEMM but trace has zero Cube ops → definitely invalid
    if formula_gemm > 0 and trace_cube == 0:
        return False, (f"FORMULA_TRACE_MISMATCH: formula declares {formula_gemm} "
                       f"GEMM steps but trace has 0 Cube instructions")

    # If trace Cube ops are less than 50% of declared GEMM steps → suspicious
    # (tiling can produce MORE Cube ops than formula steps, but never fewer
    #  by more than a factor of 2 due to small-tile merging)
    if formula_gemm > 0 and trace_cube < formula_gemm * 0.5:
        return False, (f"FORMULA_TRACE_MISMATCH: formula declares {formula_gemm} "
                       f"GEMM steps but trace has only {trace_cube} Cube ops "
                       f"(ratio {trace_cube / max(formula_gemm, 1):.2f}x < 0.5x threshold)")

    return True, ""


# ── Scoring ────────────────────────────────────────────────────────────────

CHOLESKY_BLOCK_BASELINE = {
    16: {"finish_cycle": 2828, "cube_util": 1.80, "avg_parallelism": 0.908},
    32: {"finish_cycle": 15525, "cube_util": 1.80, "avg_parallelism": 0.800},
    64: {"finish_cycle": 104628, "cube_util": 1.80, "avg_parallelism": 0.750},
}


def compute_score(algo: AlgorithmIdentity, trace: Dict[str, float],
                  rel_error: float, u: int,
                  threshold: float = 0.01) -> Tuple[float, Dict, bool]:
    details = {"algo_name": algo.name, "block_size": algo.block_size,
               "layers": algo.layers, "rel_error": round(rel_error, 8),
               "finish_cycle": int(trace["finish_cycle"]),
               "cube_util": round(trace["cube_util"], 4),
               "avg_parallelism": round(trace.get("avg_parallelism", 0), 4),
               "is_valid": True}

    if rel_error > threshold:
        details["is_valid"] = False
        details["reason"] = f"REL_ERROR_EXCEEDED ({rel_error:.2e}>{threshold:.0e})"
        return -float("inf"), details, False

    bl = CHOLESKY_BLOCK_BASELINE.get(u, {"finish_cycle": 2828, "cube_util": 1.80, "avg_parallelism": 0.908})
    s_cycle = bl["finish_cycle"] / max(trace["finish_cycle"], 1)
    # Cap cube ratio at 3.0 to prevent extreme CU values (e.g. LDL's 57%/1.8%=31.8x)
    # from dominating the score. Cube utilization is an efficiency metric, not a
    # correctness metric — high CU with similar cycles should not produce a 5x score gap.
    raw_cube = trace["cube_util"] / max(bl["cube_util"], 0.01)
    s_cube = min(raw_cube, 3.0)
    s_parallel = trace.get("avg_parallelism", 0) / max(bl["avg_parallelism"], 0.01)
    # Weights: cycle=0.80, cube=0.10, pipeline parallelism=0.10
    # Cycle is the primary metric; Cube and parallelism are efficiency modifiers.
    score = 0.80 * s_cycle + 0.10 * s_cube + 0.10 * s_parallel
    details["cycle_ratio"] = round(s_cycle, 4)
    details["cube_ratio"] = round(s_cube, 4)
    details["parallel_ratio"] = round(s_parallel, 4)
    details["raw_score"] = round(score, 4)
    return score, details, True


# ── Main entry ─────────────────────────────────────────────────────────────

def score_operator(formula_json: str, trace_csv: str, nr: int, nt: int,
                   snr_db_list: List[float] = None) -> dict:
    if snr_db_list is None:
        snr_db_list = [0.0, 5.0, 10.0, 15.0, 20.0]

    algo = identify_algorithm(formula_json)
    algo = refine_from_trace(algo, trace_csv)
    trace = extract_trace_stats(trace_csv)

    # Cross-validate: ensure trace actually executed the declared operations
    cv_ok, cv_reason = validate_trace_formula_consistency(formula_json, trace_csv)
    if not cv_ok:
        return {"score": None, "is_valid": False,
                "details": {"error": cv_reason, "is_valid": False,
                            "finish_cycle": int(trace.get("finish_cycle", 0)),
                            "cube_util": trace.get("cube_util", 0)}}

    rng = np.random.default_rng(42)
    rel_errors = []
    for snr_db in snr_db_list:
        snr_lin = 10.0 ** (snr_db / 10.0)
        noise_var = 1.0 / snr_lin
        h = (rng.standard_normal((nr, nt)) + 1j * rng.standard_normal((nr, nt))) / np.sqrt(2)
        a_mat = h.conj().T @ h + noise_var * np.eye(nt)
        a_ref = np.linalg.inv(a_mat)
        a_cand = compute_reference_inverse(a_mat, algo, formula_json)
        rel_errors.append(float(np.linalg.norm(a_cand - a_ref, 'fro')
                                / max(np.linalg.norm(a_ref, 'fro'), 1e-12)))

    score, details, valid = compute_score(algo, trace, float(np.mean(rel_errors)), nt)
    details["snr_db_list"] = snr_db_list
    details["rel_errors_per_snr"] = [round(e, 6) for e in rel_errors]
    details["avg_rel_error"] = round(float(np.mean(rel_errors)), 8)
    return {"score": score, "is_valid": valid, "details": details}


def main():
    p = argparse.ArgumentParser(description="UOBS Black-Box Scorer (Phase 2)")
    p.add_argument("--formula", required=True)
    p.add_argument("--trace", required=True)
    p.add_argument("--nr", type=int, default=64)
    p.add_argument("--nt", type=int, default=16)
    p.add_argument("--snr-db", type=str, default="0,5,10,15,20")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    result = score_operator(args.formula, args.trace, args.nr, args.nt,
                            [float(x) for x in args.snr_db.split(",")])
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        d = result["details"]
        print(f"Score: {result['score']}")
        print(f"  Algorithm: {d['algo_name']} (B={d['block_size']}, L={d['layers']})")
        print(f"  Valid: {d['is_valid']} | Rel error: {d['avg_rel_error']}")
        print(f"  Cycle: {d['finish_cycle']} | Cube util: {d['cube_util']}%")
        print(f"  Cycle ratio: {d.get('cycle_ratio','N/A')}x (vs Chol-Block)")
        if not d["is_valid"]:
            print(f"  FAIL: {d.get('reason','?')}")


if __name__ == "__main__":
    main()
