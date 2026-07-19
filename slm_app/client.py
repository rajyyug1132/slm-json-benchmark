"""Ollama HTTP client with streaming metrics and schema-validated retries.

All timing is wall-clock on the client side except tokens/sec, which uses
Ollama's own eval_count / eval_duration counters from the final stream chunk.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import requests
from pydantic import ValidationError

from .schema import JSON_SCHEMA_INSTRUCTION, SLMResponse

OLLAMA_URL = "http://localhost:11434"
MAX_RETRIES = 2  # total attempts = 1 + MAX_RETRIES


@dataclass
class Attempt:
    """Raw measurements for one generation attempt."""

    raw_text: str = ""
    ttft_s: float | None = None          # time to first streamed token
    latency_s: float = 0.0               # full wall-clock for this attempt
    eval_count: int = 0                  # output tokens (Ollama counter)
    eval_duration_ns: int = 0            # output generation time (Ollama counter)
    prompt_eval_duration_ns: int = 0
    json_valid: bool = False             # parses as JSON at all
    schema_valid: bool = False           # passes SLMResponse validation
    error: str | None = None

    @property
    def tokens_per_s(self) -> float | None:
        if self.eval_duration_ns > 0:
            return self.eval_count / (self.eval_duration_ns / 1e9)
        return None


@dataclass
class AskResult:
    """Outcome of ask(): the validated response (if any) plus every attempt."""

    response: SLMResponse | None
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.response is not None

    @property
    def first_try_valid(self) -> bool:
        return bool(self.attempts) and self.attempts[0].schema_valid

    @property
    def total_latency_s(self) -> float:
        return sum(a.latency_s for a in self.attempts)


def _strip_fences(text: str) -> str:
    """Models love ```json fences despite instructions; strip them before parsing."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def generate_stream(model: str, messages: list[dict], temperature: float = 0.0,
                    timeout_s: float = 300.0) -> Attempt:
    """One streamed /api/chat call, measuring TTFT and pulling Ollama's counters."""
    attempt = Attempt()
    start = time.perf_counter()
    try:
        with requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": True,
                "options": {"temperature": temperature},
            },
            stream=True,
            timeout=timeout_s,
        ) as resp:
            resp.raise_for_status()
            chunks: list[str] = []
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if attempt.ttft_s is None and data.get("message", {}).get("content"):
                    attempt.ttft_s = time.perf_counter() - start
                chunks.append(data.get("message", {}).get("content", ""))
                if data.get("done"):
                    attempt.eval_count = data.get("eval_count", 0)
                    attempt.eval_duration_ns = data.get("eval_duration", 0)
                    attempt.prompt_eval_duration_ns = data.get("prompt_eval_duration", 0)
            attempt.raw_text = "".join(chunks)
    except requests.RequestException as e:
        attempt.error = f"{type(e).__name__}: {e}"
    attempt.latency_s = time.perf_counter() - start

    if attempt.error is None:
        cleaned = _strip_fences(attempt.raw_text)
        try:
            parsed = json.loads(cleaned)
            attempt.json_valid = True
            SLMResponse.model_validate(parsed)
            attempt.schema_valid = True
        except json.JSONDecodeError as e:
            attempt.error = f"invalid JSON: {e}"
        except ValidationError as e:
            attempt.error = f"schema violation: {e.errors()[0].get('msg', 'invalid')}"
    return attempt


def ask(model: str, question: str, temperature: float = 0.0,
        max_retries: int = MAX_RETRIES) -> AskResult:
    """Ask with schema enforcement. On invalid output, reprompt with the error."""
    messages = [
        {"role": "system", "content": JSON_SCHEMA_INSTRUCTION},
        {"role": "user", "content": question},
    ]
    result = AskResult(response=None)
    for _ in range(1 + max_retries):
        attempt = generate_stream(model, messages, temperature)
        result.attempts.append(attempt)
        if attempt.schema_valid:
            result.response = SLMResponse.model_validate(
                json.loads(_strip_fences(attempt.raw_text)))
            return result
        # feed the failure back so the model can correct itself
        messages.append({"role": "assistant", "content": attempt.raw_text or "(empty)"})
        messages.append({
            "role": "user",
            "content": (
                f"Your previous reply was rejected: {attempt.error}. "
                "Reply again with ONLY the JSON object, exactly matching the schema."
            ),
        })
    return result


def loaded_model_memory(model: str) -> dict | None:
    """Return Ollama's /api/ps entry for a loaded model (size, size_vram in bytes)."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/ps", timeout=10)
        r.raise_for_status()
        for m in r.json().get("models", []):
            if m.get("name", "").startswith(model) or m.get("model", "").startswith(model):
                return {"size": m.get("size", 0), "size_vram": m.get("size_vram", 0)}
    except requests.RequestException:
        pass
    return None
