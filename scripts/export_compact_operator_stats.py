#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


DETAIL_FILE_NAMES = {"detailed_cycles_v3.csv", "block_jacobi_cycle_detail.csv"}


@dataclass
class Bucket:
    operator_mode: str
    event_group: str
    compute_op: str
    cycles: List[float]
    event_keys: set[str]
    formulas: set[str]


def normalize_event_key(event_key: str) -> str:
    text = event_key.strip()
    text = re.sub(r"PACK\d+", "PACK*", text)
    text = re.sub(r"\d+", "*", text)
    return text


def normalize_formula(formula: str) -> str:
    text = " ".join(formula.split())
    text = re.sub(r"\d+", "n", text)
    return text


def discover_input_csvs(root: Path) -> List[Path]:
    return sorted([path for path in root.rglob("*.csv") if path.name in DETAIL_FILE_NAMES])


def to_compact_path(input_csv: Path) -> Path:
    return input_csv.with_name(f"{input_csv.stem}_compact_stats.csv")


def aggregate_one_file(input_csv: Path) -> Tuple[Path, int, int]:
    buckets: Dict[Tuple[str, str, str], Bucket] = {}

    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            operator_mode = str(row.get("operator_mode", "")).strip()
            event_key = str(row.get("event_key", "")).strip()
            compute_op = str(row.get("compute_op", "")).strip()
            formula = str(row.get("formula", "")).strip()
            cycles_str = str(row.get("compute_cycles", "0")).strip()

            if not event_key:
                continue

            event_group = normalize_event_key(event_key)
            cycles = float(cycles_str) if cycles_str else 0.0
            formula_tmpl = normalize_formula(formula)

            key = (operator_mode, event_group, compute_op)
            if key not in buckets:
                buckets[key] = Bucket(
                    operator_mode=operator_mode,
                    event_group=event_group,
                    compute_op=compute_op,
                    cycles=[],
                    event_keys=set(),
                    formulas=set(),
                )

            bucket = buckets[key]
            bucket.cycles.append(cycles)
            bucket.event_keys.add(event_key)
            if formula_tmpl:
                bucket.formulas.add(formula_tmpl)

    output_csv = to_compact_path(input_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    source_total_cycles = 0.0
    for bucket in buckets.values():
        values = bucket.cycles
        bucket_total = sum(values)
        source_total_cycles += bucket_total
        rows.append(
            {
                "operator_mode": bucket.operator_mode,
                "event_group": bucket.event_group,
                "compute_op": bucket.compute_op,
                "row_count": str(len(values)),
                "event_key_count": str(len(bucket.event_keys)),
                "total_cycles": f"{bucket_total:.2f}",
                "mean_cycles": f"{statistics.mean(values):.2f}",
                "median_cycles": f"{statistics.median(values):.2f}",
                "min_cycles": f"{min(values):.2f}",
                "max_cycles": f"{max(values):.2f}",
                "share_pct": "0.00",
                "overall_total_cycles": "0.00",
                "example_event_keys": "|".join(sorted(bucket.event_keys)[:5]),
                "formula_template": "|".join(sorted(bucket.formulas)[:3]),
            }
        )

    rows.sort(key=lambda item: float(item["total_cycles"]), reverse=True)

    for row in rows:
        total = float(row["total_cycles"])
        pct = (total / source_total_cycles * 100.0) if source_total_cycles > 0 else 0.0
        row["share_pct"] = f"{pct:.2f}"
        row["overall_total_cycles"] = f"{source_total_cycles:.2f}"

    rows.append(
        {
            "operator_mode": rows[0]["operator_mode"] if rows else "",
            "event_group": "__TOTAL__",
            "compute_op": "ALL",
            "row_count": str(sum(int(row["row_count"]) for row in rows)),
            "event_key_count": str(sum(int(row["event_key_count"]) for row in rows)),
            "total_cycles": f"{source_total_cycles:.2f}",
            "mean_cycles": "",
            "median_cycles": "",
            "min_cycles": "",
            "max_cycles": "",
            "share_pct": "100.00",
            "overall_total_cycles": f"{source_total_cycles:.2f}",
            "example_event_keys": "",
            "formula_template": "",
        }
    )

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "operator_mode",
                "event_group",
                "compute_op",
                "row_count",
                "event_key_count",
                "total_cycles",
                "mean_cycles",
                "median_cycles",
                "min_cycles",
                "max_cycles",
                "share_pct",
                "overall_total_cycles",
                "example_event_keys",
                "formula_template",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return output_csv, len(rows), sum(len(bucket.cycles) for bucket in buckets.values())


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate compact operator-cycle summary CSVs.")
    parser.add_argument("--root", default="result_new", help="Root directory to scan for detailed cycle CSVs.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"root not found: {root}")

    inputs = discover_input_csvs(root)
    if not inputs:
        raise SystemExit("No detailed operator CSV found.")

    index_rows = []
    for input_csv in inputs:
        output_csv, compact_rows, source_rows = aggregate_one_file(input_csv)
        print(f"[ok] {input_csv} -> {output_csv} (source_rows={source_rows}, compact_rows={compact_rows})")
        index_rows.append(
            {
                "input_csv": str(input_csv),
                "output_csv": str(output_csv),
                "source_rows": str(source_rows),
                "compact_rows": str(compact_rows),
            }
        )

    index_path = root / "operator_compact_summary_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["input_csv", "output_csv", "source_rows", "compact_rows"],
        )
        writer.writeheader()
        writer.writerows(index_rows)

    print(f"[done] index={index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
