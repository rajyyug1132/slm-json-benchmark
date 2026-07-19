"""Run the full prompt suite against one model and write results/<model>.json.

Usage (from repo root):
    python benchmarks/benchmark.py --model llama3.2
    python benchmarks/benchmark.py --model llama3.2 --temperature 0.7

Measures per prompt: time-to-first-token, per-attempt latency, total latency
(incl. retries), tokens/sec (Ollama's own counters), JSON validity, schema
validity, retries used, and answer quality against expected substrings.
Also records model memory footprint from Ollama's /api/ps.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slm_app.client import ask, loaded_model_memory  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = Path(__file__).resolve().parent / "prompts.jsonl"
RESULTS_DIR = ROOT / "results"


def slug(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def run(model: str, temperature: float) -> dict:
    prompts = [json.loads(line) for line in PROMPTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = []
    print(f"== {model} (temperature={temperature}) — {len(prompts)} prompts ==")

    # warm-up: load the model into memory so cold-start doesn't skew prompt 1's TTFT
    print("  warm-up...", flush=True)
    ask(model, "Say hi.", temperature, max_retries=0)

    memory = None
    for i, p in enumerate(prompts, 1):
        r = ask(model, p["prompt"], temperature)
        first = r.attempts[0]
        quality = None
        if r.ok and p.get("expect_any"):
            ans = r.response.answer.lower()
            quality = any(exp.lower() in ans for exp in p["expect_any"])
        rec = {
            "id": p["id"],
            "category": p["category"],
            "ok": r.ok,
            "first_try_valid": r.first_try_valid,
            "json_valid_first": first.json_valid,
            "attempts": len(r.attempts),
            "ttft_s": round(first.ttft_s, 4) if first.ttft_s is not None else None,
            "first_attempt_latency_s": round(first.latency_s, 4),
            "total_latency_s": round(r.total_latency_s, 4),
            "tokens_per_s": round(first.tokens_per_s, 2) if first.tokens_per_s else None,
            "eval_tokens": first.eval_count,
            "quality_pass": quality,
            "error": None if r.ok else r.attempts[-1].error,
        }
        records.append(rec)
        if memory is None:
            memory = loaded_model_memory(model)
        status = "ok " if r.ok else "FAIL"
        q = {True: "hit", False: "miss", None: "-"}[quality]
        print(f"  [{i:2}/{len(prompts)}] {p['id']:8} {status} attempts={rec['attempts']} "
              f"lat={rec['total_latency_s']:.1f}s q={q}")

    n = len(records)
    ok_recs = [r for r in records if r["ok"]]
    retried = [r for r in records if not r["first_try_valid"]]
    retried_ok = [r for r in retried if r["ok"]]
    quality_scored = [r for r in records if r["quality_pass"] is not None]
    ttfts = [r["ttft_s"] for r in ok_recs if r["ttft_s"] is not None]
    tps = [r["tokens_per_s"] for r in ok_recs if r["tokens_per_s"]]

    summary = {
        "model": model,
        "temperature": temperature,
        "n_prompts": n,
        "schema_valid_first_try_rate": round(sum(r["first_try_valid"] for r in records) / n, 4),
        "json_valid_first_try_rate": round(sum(r["json_valid_first"] for r in records) / n, 4),
        "final_success_rate": round(len(ok_recs) / n, 4),
        "retry_success_rate": round(len(retried_ok) / len(retried), 4) if retried else None,
        "n_retried": len(retried),
        "quality_rate": round(sum(r["quality_pass"] for r in quality_scored) / len(quality_scored), 4) if quality_scored else None,
        "avg_ttft_s": round(statistics.mean(ttfts), 3) if ttfts else None,
        "median_ttft_s": round(statistics.median(ttfts), 3) if ttfts else None,
        "avg_total_latency_s": round(statistics.mean([r["total_latency_s"] for r in records]), 3),
        "median_first_attempt_latency_s": round(statistics.median([r["first_attempt_latency_s"] for r in records]), 3),
        "avg_tokens_per_s": round(statistics.mean(tps), 1) if tps else None,
        "memory_bytes": memory,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return {"summary": summary, "records": records}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    out = run(args.model, args.temperature)
    RESULTS_DIR.mkdir(exist_ok=True)
    suffix = "" if args.temperature == 0.0 else f"_t{args.temperature}"
    path = RESULTS_DIR / f"{slug(args.model)}{suffix}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    s = out["summary"]
    print(f"\nsummary: first-try schema validity {s['schema_valid_first_try_rate']:.0%}, "
          f"final success {s['final_success_rate']:.0%}, "
          f"avg tok/s {s['avg_tokens_per_s']}, wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
