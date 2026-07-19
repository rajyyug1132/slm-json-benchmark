"""Aggregate all results/*.json into results/summary.csv and print a table.

Usage: python benchmarks/compare.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

COLUMNS = [
    "model", "temperature", "n_prompts",
    "schema_valid_first_try_rate", "json_valid_first_try_rate",
    "final_success_rate", "retry_success_rate", "n_retried", "quality_rate",
    "avg_ttft_s", "median_ttft_s", "avg_total_latency_s",
    "median_first_attempt_latency_s", "avg_tokens_per_s", "memory_mb",
]


def load_summaries() -> list[dict]:
    rows = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        if "summary" not in data:
            continue
        s = dict(data["summary"])
        mem = s.pop("memory_bytes", None)
        s["memory_mb"] = round(mem["size"] / 1e6) if mem and mem.get("size") else None
        rows.append(s)
    rows.sort(key=lambda r: (-(r["schema_valid_first_try_rate"] or 0),
                             r["avg_total_latency_s"] or 9e9))
    return rows


def main() -> int:
    rows = load_summaries()
    if not rows:
        print("no results found in results/ — run benchmarks/benchmark.py first")
        return 1

    out = RESULTS_DIR / "summary.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    fmt = "{:<22} {:>6} {:>10} {:>9} {:>9} {:>8} {:>8} {:>9}"
    print(fmt.format("model", "temp", "schema-1st", "final-ok", "retry-ok", "tok/s", "ttft", "mem-MB"))
    for r in rows:
        print(fmt.format(
            r["model"], r["temperature"],
            f"{r['schema_valid_first_try_rate']:.0%}",
            f"{r['final_success_rate']:.0%}",
            f"{r['retry_success_rate']:.0%}" if r["retry_success_rate"] is not None else "-",
            r["avg_tokens_per_s"] or "-",
            f"{r['avg_ttft_s']}s" if r["avg_ttft_s"] else "-",
            r["memory_mb"] or "-",
        ))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
