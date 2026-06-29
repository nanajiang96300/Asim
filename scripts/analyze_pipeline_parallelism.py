#!/usr/bin/env python3
"""
Pipeline parallelism and utilization analyzer.
Reads trace CSV, computes:
  - Per-cycle pipeline occupancy (which units are active)
  - Average/peak parallelism (number of units active per cycle)
  - Proper utilization ratios: compute cycles and memory cycles
    separated (since they run on different hardware and can overlap)
  - Pipeline overlap score for UOBS scorer
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def analyze_trace(trace_path: str, core_id: int = 0) -> dict:
    """
    Analyze trace CSV and compute pipeline parallelism and utilization metrics.

    Returns a dict with:
      - core_max_end: last cycle for this core
      - global_max_end: last cycle across all cores
      - avg_parallelism: average number of units active per cycle
      - peak_parallelism: max units active in a single cycle
      - parallelism_distribution: {n: count} for n units active
      - compute_active_ratio: fraction of span where any compute unit is active
      - memory_active_ratio: fraction of span where any memory unit is active
      - per_unit_active_ratio: {unit: active_cycles/span}
      - pipeline_overlap_pct: cycles with ≥2 active units / total span
    """
    core_prefix = f"Core{core_id}_"

    # Read all events
    events = []  # (unit, start, end)
    global_max = 0
    core_max = 0

    with open(trace_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            unit = row["unit"].strip('"')
            start = int(row["start_cycle"])
            end = int(row["end_cycle"])

            if end > global_max:
                global_max = end

            if unit.startswith(core_prefix):
                unit_name = unit.split("_", 1)[1]
                events.append((unit_name, start, end))
                if end > core_max:
                    core_max = end

    if core_max == 0:
        return {"error": "no events found"}

    span = core_max

    # Map unit names to categories
    compute_units = {"Cube", "Vector", "Scalar"}
    memory_units = {"MTE2", "MTE3"}
    all_categories = compute_units | memory_units | {"Wait"}

    # Build per-cycle activity mask
    # mask bit: Cube=1, Vector=2, Scalar=4, MTE2=8, MTE3=16, Wait=32
    unit_to_bit = {
        "Cube": 1, "Vector": 2, "Scalar": 4,
        "MTE2": 8, "MTE3": 16, "Wait": 32,
        "Cube_Wait": 64, "Vector_Wait": 128,
    }
    bit_to_unit = {v: k for k, v in unit_to_bit.items()}

    # Use a dense array for cycles
    # For large spans (millions), use sparse approach
    if span > 500_000:
        return _analyze_sparse(events, span, core_max, global_max)

    active_mask = [0] * span

    for unit, start, end in events:
        if unit in unit_to_bit:
            bit = unit_to_bit[unit]
            for c in range(start, min(end, span)):
                active_mask[c] |= bit

    # Compute statistics
    parallelism_counts: Dict[int, int] = defaultdict(int)
    compute_active = 0
    memory_active = 0
    unit_active: Dict[str, int] = {}

    for bit_name in ["Cube", "Vector", "Scalar", "MTE2", "MTE3", "Wait"]:
        unit_active[bit_name] = 0

    for mask in active_mask:
        # Count how many bits are set (excluding Wait)
        is_compute = (mask & 1) or (mask & 2) or (mask & 4)  # Cube|Vector|Scalar
        is_memory = (mask & 8) or (mask & 16)  # MTE2|MTE3

        if is_compute:
            compute_active += 1
        if is_memory:
            memory_active += 1

        # Count non-Wait active units
        active_units = 0
        for bit_name, bit in [("Cube", 1), ("Vector", 2), ("Scalar", 4),
                              ("MTE2", 8), ("MTE3", 16)]:
            if mask & bit:
                unit_active[bit_name] += 1
                active_units += 1

        # Wait is tracked separately
        if mask & 32:
            unit_active["Wait"] += 1

        parallelism_counts[active_units] += 1

    total_cycle_units = sum(count * n for n, count in parallelism_counts.items())
    total_cycles_with_units = sum(parallelism_counts.values())

    avg_parallelism = total_cycle_units / total_cycles_with_units if total_cycles_with_units > 0 else 0
    peak_parallelism = max(parallelism_counts.keys()) if parallelism_counts else 0

    # Cycles with ≥2 active compute/memory units = pipeline overlap
    overlap_cycles = sum(count for n, count in parallelism_counts.items() if n >= 2)
    pipeline_overlap_pct = (overlap_cycles / span) * 100 if span > 0 else 0

    result = {
        "core_max_end": core_max,
        "global_max_end": global_max,
        "span_cycles": span,
        "avg_parallelism": round(avg_parallelism, 4),
        "peak_parallelism": peak_parallelism,
        "parallelism_distribution": dict(sorted(parallelism_counts.items())),
        "compute_active_ratio": round(compute_active / span * 100, 2) if span > 0 else 0,
        "memory_active_ratio": round(memory_active / span * 100, 2) if span > 0 else 0,
        "per_unit_active_ratio": {
            k: round(v / span * 100, 2) if span > 0 else 0
            for k, v in unit_active.items()
        },
        "pipeline_overlap_pct": round(pipeline_overlap_pct, 2),
    }

    return result


def _analyze_sparse(events: list, span: int, core_max: int, global_max: int) -> dict:
    """Fallback for very long traces: use event-level statistics."""
    # For sparse analysis, we compute overlap of time intervals
    # Group events by unit type and count non-overlapping total duration
    from collections import defaultdict

    unit_intervals = defaultdict(list)
    for unit, start, end in events:
        if unit in ("Cube", "Vector", "Scalar", "MTE2", "MTE3", "Wait"):
            unit_intervals[unit].append((start, end))

    # Merge overlapping intervals per unit
    unit_total = {}
    unit_count = {}
    for unit, intervals in unit_intervals.items():
        intervals.sort()
        merged = []
        for s, e in intervals:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        total = sum(e - s for s, e in merged)
        unit_total[unit] = total
        unit_count[unit] = len(intervals)

    # Compute maximum potential parallelism by finding time intervals
    # where multiple units have overlapping events
    all_points = []
    for unit, intervals in unit_intervals.items():
        for s, e in intervals:
            all_points.append((s, 1, unit))
            all_points.append((e, -1, unit))

    all_points.sort()

    active_units = set()
    parallelism_per_point = defaultdict(int)
    max_parallel = 0
    prev_cycle = 0

    for cycle, delta, unit in all_points:
        if cycle > prev_cycle and active_units:
            par = len(active_units)
            parallelism_per_point[par] += cycle - prev_cycle
            if par > max_parallel:
                max_parallel = par
        if delta == 1:
            active_units.add(unit)
        else:
            active_units.discard(unit)
        prev_cycle = cycle

    total_cycle_units = sum(n * c for n, c in parallelism_per_point.items())
    total_cycles = sum(parallelism_per_point.values()) or 1

    result = {
        "core_max_end": core_max,
        "global_max_end": global_max,
        "span_cycles": core_max,
        "avg_parallelism": round(total_cycle_units / total_cycles, 4),
        "peak_parallelism": max_parallel,
        "parallelism_distribution": dict(parallelism_per_point),
        "per_unit_total_duration": {k: round(v, 0) for k, v in unit_total.items()},
        "per_unit_event_count": unit_count,
        "pipeline_overlap_pct": round(
            sum(c for n, c in parallelism_per_point.items() if n >= 2) / core_max * 100, 2
        ) if core_max > 0 else 0,
        "_sparse": True,
    }

    return result


def format_report(result: dict) -> str:
    """Format results for document inclusion."""
    lines = []
    if "error" in result:
        return f"Error: {result['error']}"

    lines.append(f"Span: {result['span_cycles']} cycles")
    lines.append(f"Avg Parallelism: {result['avg_parallelism']:.2f}")
    lines.append(f"Peak Parallelism: {result['peak_parallelism']}")
    lines.append(f"Pipeline Overlap (≥2 units): {result['pipeline_overlap_pct']:.1f}%")

    if "compute_active_ratio" in result:
        lines.append(f"Compute Active Ratio: {result['compute_active_ratio']:.1f}%")
        lines.append(f"Memory Active Ratio: {result['memory_active_ratio']:.1f}%")

    lines.append("Per-unit active ratio:")
    if "per_unit_active_ratio" in result:
        for unit, pct in sorted(result["per_unit_active_ratio"].items()):
            lines.append(f"  {unit}: {pct:.1f}%")
    elif "per_unit_total_duration" in result:
        span = result["span_cycles"]
        for unit, dur in sorted(result["per_unit_total_duration"].items()):
            lines.append(f"  {unit}: {dur/span*100:.1f}% ({dur:.0f} cycles)")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline parallelism analyzer")
    parser.add_argument("trace", help="Path to trace CSV")
    parser.add_argument("--core", type=int, default=0, help="Core ID (default: 0)")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    result = analyze_trace(args.trace, args.core)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_report(result))


if __name__ == "__main__":
    main()
