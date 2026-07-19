"""Command-line assistant: python -m slm_app.cli "question" --model llama3.2"""
import argparse
import json
import sys

from .client import ask


def main() -> int:
    p = argparse.ArgumentParser(description="Offline SLM assistant (schema-enforced JSON)")
    p.add_argument("question")
    p.add_argument("--model", default="llama3.2")
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args()

    result = ask(args.model, args.question, args.temperature)
    if not result.ok:
        last = result.attempts[-1]
        print(f"FAILED after {len(result.attempts)} attempts: {last.error}", file=sys.stderr)
        return 1

    print(json.dumps(result.response.model_dump(), indent=2))
    a = result.attempts[-1]
    tps = f"{a.tokens_per_s:.1f}" if a.tokens_per_s else "n/a"
    print(
        f"[{args.model}] attempts={len(result.attempts)} "
        f"ttft={a.ttft_s:.2f}s latency={result.total_latency_s:.2f}s tok/s={tps}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
