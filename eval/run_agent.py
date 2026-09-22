"""Run the Strands ticket agent once over every dataset ticket and freeze the output.

Usage: python -m eval.run_agent

Writes eval/results.json: [{id, question, answer, kb_evidence, tool_calls, expected_behavior}, ...]

This is frozen input for both judges - run this exactly once per "agent version" you
want to evaluate. Re-running it regenerates agent answers and reintroduces agent-side
variance into what repeatability.py and human_agreement.py treat as judge-only variance.
Resumable: tickets already present in results.json are skipped unless --force is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.common import ROOT, load_env, log_call, retry_with_backoff

DATASET_PATH = ROOT / "eval" / "dataset.json"
RESULTS_PATH = ROOT / "eval" / "results.json"


def load_dataset() -> list[dict]:
    with open(DATASET_PATH) as f:
        return json.load(f)


def load_existing_results() -> dict[str, dict]:
    if not RESULTS_PATH.exists():
        return {}
    with open(RESULTS_PATH) as f:
        return {r["id"]: r for r in json.load(f)}


def save_results(results_by_id: dict[str, dict]) -> None:
    ordered = [results_by_id[k] for k in results_by_id]
    RESULTS_PATH.write_text(json.dumps(ordered, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-run tickets even if already in results.json")
    args = parser.parse_args()

    load_env()
    # Import after env is loaded so ticket_agent's require_env() calls succeed.
    from agent.ticket_agent import run_ticket

    dataset = load_dataset()
    existing = {} if args.force else load_existing_results()

    for ticket in dataset:
        ticket_id = ticket["id"]
        if ticket_id in existing:
            print(f"[skip] {ticket_id} already in results.json", file=sys.stderr)
            continue

        print(f"[run]  {ticket_id}: {ticket['question'][:60]!r}", file=sys.stderr)
        try:
            outcome = retry_with_backoff(
                lambda: run_ticket(ticket["question"]),
                on_retry=lambda attempt, exc, delay: log_call(
                    {"stage": "run_agent", "ticket_id": ticket_id, "event": "retry", "attempt": attempt, "error": str(exc), "delay_s": delay}
                ),
            )
        except Exception as exc:  # noqa: BLE001
            log_call({"stage": "run_agent", "ticket_id": ticket_id, "event": "error", "error": str(exc)})
            print(f"[FAIL] {ticket_id}: {exc}", file=sys.stderr)
            # Save progress so far before bailing - no silent data loss on a partial run.
            save_results(existing)
            raise

        record = {
            "id": ticket_id,
            "question": ticket["question"],
            "answer": outcome["answer"],
            "kb_evidence": outcome["kb_evidence"],
            "tool_calls": outcome["tool_calls"],
            "expected_behavior": ticket.get("expected_behavior", ""),
        }
        log_call({"stage": "run_agent", "ticket_id": ticket_id, "event": "success"})
        existing[ticket_id] = record
        save_results(existing)  # persist after every ticket - resumable, no silent data loss

    print(f"Done. {len(existing)} tickets in {RESULTS_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
