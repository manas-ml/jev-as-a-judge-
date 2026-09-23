"""Jev judge: one structured-decision call per ticket via OpenRouter's System One API.

Verified against the installed `openrouter` Python SDK (v1.2.19), which ships a
`system_one.create(model=, questions=, state=)` client returning a `DecisionsResponse`
with `answers: {name: NoulAnswer|ChoiceAnswer|ScoreAnswer}` and `usage: {input_tokens,
output_tokens, cost}`. This matches the REST contract in the spec
(`POST /api/v1/systemone`) - the SDK is just a typed wrapper around that same endpoint.

We still parse defensively (never assume a field is present beyond what the SDK's own
schema guarantees) because this is a live third-party API and the spec explicitly warns
not to hard-code an unseen response shape.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from openrouter import OpenRouter

from eval.common import (
    compute_quality,
    derive_does_pass,
    log_call,
    require_env,
    retry_with_backoff,
)

# Fixed question set sent in a single Decisions call per ticket-evaluation.
# `noul` = a continuous 0-1 answer; `choice` = a labeled option with confidence/probabilities.
QUESTIONS = {
    "groundedness": {
        "type": "noul",
        "instructions": "Is the answer fully supported by kb_evidence, with no unsupported claims?",
    },
    "search_behavior": {
        "type": "choice",
        "instructions": "How well did the agent's KB search match the ticket?",
        "criteria": {
            "appropriate": "Searched the KB and used relevant results.",
            "unnecessary": "Searched when the question needed no KB lookup.",
            "failed": "Should have searched but didn't, or searched and used irrelevant results.",
        },
    },
    "usefulness": {
        "type": "noul",
        "instructions": "Does the answer satisfy expected_behavior and actually resolve or correctly escalate the ticket?",
    },
}

_client: OpenRouter | None = None
_client_lock = threading.Lock()


def _get_client() -> OpenRouter:
    # Double-checked locking: evaluate() is called concurrently by repeatability.py's
    # thread pool, and the first calls to land could otherwise race to construct
    # multiple clients before the module-level singleton is set.
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = OpenRouter(api_key=require_env("OPENROUTER_API_KEY"))
    return _client


def _build_state(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": case["question"],
        "answer": case["answer"],
        "kb_evidence": [
            f"{e.get('title', e.get('id', ''))}: {e.get('text', '')}" for e in case.get("kb_evidence", [])
        ],
        "expected_behavior": case.get("expected_behavior", ""),
    }


def _extract_choice_answer(answer: Any) -> dict[str, Any]:
    """Defensively pull choice/confidence/probabilities off a choice-type answer."""
    choice = getattr(answer, "choice", None)
    confidence = getattr(answer, "confidence", None)
    probabilities = getattr(answer, "probabilities", None)
    return {"choice": choice, "confidence": confidence, "probabilities": probabilities}


def evaluate(case: dict[str, Any], model_id: str | None = None) -> dict[str, Any]:
    """Run one Jev Decisions call for a single ticket-evaluation.

    `case` must have: question, answer, kb_evidence (list of KB entry dicts), expected_behavior.
    Returns a dict with the raw per-question answers, the derived composite quality/does_pass,
    usage (input_tokens/output_tokens/cost), and latency_s.
    """
    client = _get_client()
    model_id = model_id or require_env("JEV_MODEL_ID")
    state = _build_state(case)

    start = time.monotonic()

    def _call():
        return client.system_one.create(model=model_id, questions=QUESTIONS, state=state)

    try:
        response = retry_with_backoff(
            _call,
            on_retry=lambda attempt, exc, delay: log_call(
                {
                    "judge": "jev",
                    "ticket_id": case.get("id"),
                    "event": "retry",
                    "attempt": attempt,
                    "error": str(exc),
                    "delay_s": delay,
                }
            ),
        )
    except Exception as exc:  # noqa: BLE001
        latency_s = time.monotonic() - start
        log_call(
            {
                "judge": "jev",
                "ticket_id": case.get("id"),
                "event": "error",
                "error": str(exc),
                "latency_s": latency_s,
            }
        )
        raise

    latency_s = time.monotonic() - start

    answers = response.answers
    groundedness_ans = answers.get("groundedness")
    search_behavior_ans = answers.get("search_behavior")
    usefulness_ans = answers.get("usefulness")

    groundedness = getattr(groundedness_ans, "noul", None)
    usefulness = getattr(usefulness_ans, "noul", None)
    search_choice_info = _extract_choice_answer(search_behavior_ans)
    search_behavior = search_choice_info["choice"]

    if groundedness is None or usefulness is None or search_behavior is None:
        raise ValueError(
            f"Jev response missing expected answer fields for ticket {case.get('id')}: {answers!r}"
        )

    quality = compute_quality(groundedness, search_behavior, usefulness)
    does_pass = derive_does_pass(quality)

    usage = response.usage
    record = {
        "judge": "jev",
        "ticket_id": case.get("id"),
        "model": response.model,
        "groundedness": groundedness,
        "search_behavior": search_behavior,
        "search_behavior_confidence": search_choice_info["confidence"],
        "search_behavior_probabilities": search_choice_info["probabilities"],
        "usefulness": usefulness,
        "quality": quality,
        "does_pass": does_pass,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost": usage.cost,
        "latency_s": latency_s,
        "response_id": response.id,
    }
    log_call({"judge": "jev", "ticket_id": case.get("id"), "event": "success", **record})
    return record


if __name__ == "__main__":
    # Smoke test: `python -m eval.judges.jev_judge` runs one live evaluation.
    import json
    import sys

    from eval.common import load_env

    load_env()
    sample_case = {
        "id": "smoke",
        "question": "I was charged twice for my subscription this month, can you fix that?",
        "answer": "I checked and this looks like a duplicate charge from a retried payment. "
        "It should auto-refund within 5 business days; if it's been longer, you can open "
        "a billing dispute from Account > Billing > Dispute a charge.",
        "kb_evidence": [
            {
                "id": "kb_001",
                "title": "Duplicate charge",
                "text": "If you see two charges for the same billing period, this is usually "
                "caused by a retried payment after a temporary card decline. Duplicate charges "
                "are automatically detected and refunded within 5 business days.",
            }
        ],
        "expected_behavior": "Explain auto-refund within 5 business days and the dispute path if longer.",
    }
    try:
        result = evaluate(sample_case)
        print(json.dumps(result, indent=2, default=str))
    except Exception as exc:  # noqa: BLE001
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        sys.exit(1)
