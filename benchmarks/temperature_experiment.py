"""Temperature 0 vs 0.7: schema validity and output determinism.

Runs a 10-prompt subset 3 times at each temperature and reports, per model:
  - schema validity rate at each temperature
  - determinism: fraction of prompts whose 3 answers are byte-identical

Usage: python benchmarks/temperature_experiment.py --model llama3.2
Writes results/temperature_<model>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slm_app.client import ask  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = Path(__file__).resolve().parent / "prompts.jsonl"
SUBSET = 10
RUNS = 3
TEMPS = (0.0, 0.7)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    prompts = [json.loads(l) for l in PROMPTS.read_text(encoding="utf-8").splitlines() if l.strip()][:SUBSET]
    out = {"model": args.model, "runs": RUNS, "subset": SUBSET, "temperatures": {}}

    for temp in TEMPS:
        valid = 0
        identical = 0
        total = 0
        print(f"== temperature {temp} ==")
        for p in prompts:
            answers = []
            for _ in range(RUNS):
                r = ask(args.model, p["prompt"], temperature=temp, max_retries=0)
                total += 1
                if r.ok:
                    valid += 1
                    answers.append(r.response.answer)
                else:
                    answers.append(None)
            if len(set(answers)) == 1 and answers[0] is not None:
                identical += 1
            print(f"  {p['id']}: valid {sum(a is not None for a in answers)}/{RUNS}, "
                  f"identical={len(set(answers)) == 1}")
        out["temperatures"][str(temp)] = {
            "schema_valid_rate": round(valid / total, 4),
            "deterministic_prompt_rate": round(identical / len(prompts), 4),
        }

    dest = ROOT / "results" / f"temperature_{args.model.replace(':', '_')}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
