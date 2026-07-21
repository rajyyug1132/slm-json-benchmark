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


def load_categories() -> dict[str, dict[str, dict]]:
    """Per-model, per-category rates: {model: {category: {...}}}.

    Reads the per-prompt records the suite already writes, so this needs no
    extra benchmark runs. Models with no records (failure rows) are skipped.
    """
    out: dict[str, dict[str, dict]] = {}
    for f in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        records = data.get("records") or []
        if not records:
            continue
        by_cat: dict[str, dict] = {}
        for rec in records:
            c = by_cat.setdefault(rec["category"], {"n": 0, "valid": 0, "quality": 0, "scored": 0})
            c["n"] += 1
            c["valid"] += bool(rec["first_try_valid"])
            if rec["quality_pass"] is not None:
                c["scored"] += 1
                c["quality"] += bool(rec["quality_pass"])
        for c in by_cat.values():
            c["schema_valid_first_try_rate"] = round(c["valid"] / c["n"], 4)
            c["quality_rate"] = round(c["quality"] / c["scored"], 4) if c["scored"] else None
        out[data["summary"]["model"]] = by_cat
    return out


def write_category_csv(cats: dict[str, dict[str, dict]]) -> Path | None:
    if not cats:
        return None
    dest = RESULTS_DIR / "by_category.csv"
    with dest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "category", "n_prompts", "schema_valid_first_try_rate", "quality_rate"])
        for model, by_cat in cats.items():
            for cat, c in sorted(by_cat.items()):
                w.writerow([model, cat, c["n"], c["schema_valid_first_try_rate"], c["quality_rate"]])
    return dest


def print_category_table(cats: dict[str, dict[str, dict]]) -> None:
    if not cats:
        return
    categories = sorted({c for by_cat in cats.values() for c in by_cat})
    print("\nquality rate by category (schema validity in parens):")
    header = "{:<22}".format("model") + "".join(f"{c:>17}" for c in categories)
    print(header)
    for model, by_cat in cats.items():
        cells = []
        for cat in categories:
            c = by_cat.get(cat)
            if not c:
                cells.append(f"{'-':>17}")
                continue
            q = f"{c['quality_rate']:.0%}" if c["quality_rate"] is not None else "-"
            cells.append(f"{q} ({c['schema_valid_first_try_rate']:.0%})".rjust(17))
        print("{:<22}".format(model) + "".join(cells))


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
    cats = load_categories()
    print_category_table(cats)
    cat_csv = write_category_csv(cats)

    print(f"\nwrote {out}" + (f" and {cat_csv}" if cat_csv else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
