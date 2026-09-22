"""Claude-as-judge: one structured-output chat completion per ticket via OpenRouter.

Uses OpenRouter's OpenAI-compatible `/chat/completions` endpoint with a JSON-schema
`response_format` to force a single structured JSON object as the final output, per
the spec. One call per ticket-evaluation (not three) - this is a deliberate deviation
from the source repo, called out explicitly in report.py, since 3x the calls is most
of why that benchmark landed near $28.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from openai import OpenAI

from eval.common import log_call, require_env, retry_with_backoff

RUBRIC = """You are grading a support-ticket agent's response for a SaaS billing/account help desk.

You will be given:
- question: the customer's ticket
- answer: the agent's final response
- kb_evidence: the knowledge-base entries the agent's search actually returned (may be empty)
- expected_behavior: a human description of what a correct response looks like

Score on exactly these axes:
- groundedness (0.0-1.0): is the answer fully supported by kb_evidence, with no unsupported claims?
- search_behavior (one of "appropriate" | "unnecessary" | "failed"): "appropriate" = searched the
  KB and used relevant results; "unnecessary" = searched when the question needed no KB lookup;
  "failed" = should have searched but didn't, or searched and used irrelevant results.
- usefulness (0.0-1.0): does the answer satisfy expected_behavior and actually resolve or correctly
  escalate the ticket?
- does_pass (0 or 1): your overall pass/fail judgment for this response.
- outcome (one of "answered" | "clarification_needed" | "poor"): the practical shape of the response.

Reason briefly if you need to, but your final output must be ONLY the JSON object described by the
response schema - no prose before or after it.
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "groundedness": {"type": "number", "minimum": 0, "maximum": 1},
        "search_behavior": {"type": "string", "enum": ["appropriate", "unnecessary", "failed"]},
        "usefulness": {"type": "number", "minimum": 0, "maximum": 1},
        "does_pass": {"type": "integer", "enum": [0, 1]},
        "outcome": {"type": "string", "enum": ["answered", "clarification_needed", "poor"]},
    },
    "required": ["groundedness", "search_behavior", "usefulness", "does_pass", "outcome"],
    "additionalProperties": False,
}

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=require_env("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
    return _client


def _build_user_content(case: dict[str, Any]) -> str:
    payload = {
        "question": case["question"],
        "answer": case["answer"],
        "kb_evidence": [
            {"id": e.get("id"), "title": e.get("title"), "text": e.get("text")} for e in case.get("kb_evidence", [])
        ],
        "expected_behavior": case.get("expected_behavior", ""),
    }
    return json.dumps(payload, indent=2)


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_structured_output(message: Any) -> dict[str, Any]:
    """Defensively pull the judgment JSON out of a chat completion message.

    Tries, in order: message.content as JSON, a tool_call's function arguments as JSON,
    then a best-effort regex extraction of the first {...} block in message.content.
    Never assumes a shape the spec didn't guarantee.
    """
    content = getattr(message, "content", None)
    if content:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = _JSON_OBJECT_RE.search(content)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        for call in tool_calls:
            args = getattr(getattr(call, "function", None), "arguments", None)
            if args:
                try:
                    return json.loads(args)
                except json.JSONDecodeError:
                    continue

    raise ValueError(f"Could not extract structured JSON from Claude judge response: {message!r}")


def _validate(parsed: dict[str, Any]) -> dict[str, Any]:
    required = ["groundedness", "search_behavior", "usefulness", "does_pass", "outcome"]
    missing = [k for k in required if k not in parsed]
    if missing:
        raise ValueError(f"Claude judge response missing fields {missing}: {parsed!r}")
    if parsed["search_behavior"] not in ("appropriate", "unnecessary", "failed"):
        raise ValueError(f"Unexpected search_behavior value: {parsed['search_behavior']!r}")
    if parsed["outcome"] not in ("answered", "clarification_needed", "poor"):
        raise ValueError(f"Unexpected outcome value: {parsed['outcome']!r}")
    return parsed


def evaluate(case: dict[str, Any], model_id: str | None = None) -> dict[str, Any]:
    """Run one Claude-as-judge chat completion for a single ticket-evaluation."""
    client = _get_client()
    model_id = model_id or require_env("CLAUDE_JUDGE_MODEL_ID")

    messages = [
        {"role": "system", "content": RUBRIC},
        {"role": "user", "content": _build_user_content(case)},
    ]

    start = time.monotonic()

    def _call():
        return client.chat.completions.create(
            model=model_id,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "ticket_judgment", "strict": True, "schema": RESPONSE_SCHEMA},
            },
            extra_body={"usage": {"include": True}},
        )

    try:
        response = retry_with_backoff(
            _call,
            on_retry=lambda attempt, exc, delay: log_call(
                {
                    "judge": "claude",
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
        log_call({"judge": "claude", "ticket_id": case.get("id"), "event": "error", "error": str(exc), "latency_s": latency_s})
        raise

    latency_s = time.monotonic() - start

    parsed = _validate(_parse_structured_output(response.choices[0].message))

    usage = response.usage
    cost = getattr(usage, "cost", None) if usage is not None else None
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None

    record = {
        "judge": "claude",
        "ticket_id": case.get("id"),
        "model": response.model,
        "groundedness": float(parsed["groundedness"]),
        "search_behavior": parsed["search_behavior"],
        "usefulness": float(parsed["usefulness"]),
        "does_pass": int(parsed["does_pass"]),
        "outcome": parsed["outcome"],
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "cost": cost,
        "latency_s": latency_s,
        "response_id": response.id,
    }
    log_call({"judge": "claude", "ticket_id": case.get("id"), "event": "success", **record})
    return record


if __name__ == "__main__":
    # Smoke test: `python -m eval.judges.claude_judge` runs one live evaluation.
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
