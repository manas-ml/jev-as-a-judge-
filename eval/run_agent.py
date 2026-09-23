"""Run the Strands ticket agent once over every dataset ticket and freeze the output.

Usage:
  python -m eval.run_agent
  python -m eval.run_agent --concurrency 5   # run multiple tickets through the agent at once

Writes eval/results.json: [{id, question, answer, kb_evidence, tool_calls, expected_behavior}, ...]

This is frozen input for the judge - run this exactly once per "agent version" you
want to evaluate. Re-running it regenerates agent answers and reintroduces agent-side
variance into what repeatability.py treats as judge-only variance.
Resumable: tickets already present in results.json are skipped unless --force is passed.

Tickets run concurrently, bounded by --concurrency (default 5), via a thread pool - each
call builds its own fresh Agent (agent/ticket_agent.py:build_agent), so there's no shared
mutable state between threads beyond the read-only KB index. Unlike the serial version,
a ticket that fails after retries no longer aborts the whole run - it's logged and
skipped so the rest of the batch keeps going; re-run the same command to retry it.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from eval.common import ROOT, load_env, log_call, retry_with_backoff

DATASET_PATH = ROOT / "eval" / "dataset.json"
RESULTS_PATH = ROOT / "eval" / "results.json"

_results_lock = threading.Lock()


def load_dataset() -> list[dict]:
    with open(DATASET_PATH) as f:
        return json.load(f)


def load_existing_results() -> dict[str, dict]:
    if not RESULTS_PATH.exists():
        return {}
    with open(RESULTS_PATH) as f:
        return {r["id"]: r for r in json.load(f)}


def save_results(results_by_id: dict[str, dict], dataset_order: list[str]) -> None:
    # Written in dataset order (not completion order) so the file's row order stays
    # stable and diffable regardless of which threads finish first.
    ordered = [results_by_id[tid] for tid in dataset_order if tid in results_by_id]
    RESULTS_PATH.write_text(json.dumps(ordered, indent=2))


def _run_one(run_ticket, ticket: dict) -> tuple[str, dict | None, Exception | None]:
    ticket_id = ticket["id"]
    try:
        outcome = retry_with_backoff(
            lambda: run_ticket(ticket["question"]),
            on_retry=lambda attempt, exc, delay: log_call(
                {
                    "stage": "run_agent",
                    "ticket_id": ticket_id,
                    "event": "retry",
                    "attempt": attempt,
                    "error": str(exc),
                    "delay_s": delay,
                }
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return ticket_id, None, exc

    record = {
        "id": ticket_id,
        "question": ticket["question"],
        "answer": outcome["answer"],
        "kb_evidence": outcome["kb_evidence"],
        "tool_calls": outcome["tool_calls"],
        "expected_behavior": ticket.get("expected_behavior", ""),
    }
    return ticket_id, record, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-run tickets even if already in results.json")
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent agent runs in flight")
    args = parser.parse_args()

    load_env()
    # Import after env is loaded so ticket_agent's require_env() calls succeed.
    from agent.ticket_agent import run_ticket

    dataset = load_dataset()
    dataset_order = [t["id"] for t in dataset]
    existing = {} if args.force else load_existing_results()

    pending = [t for t in dataset if t["id"] not in existing]
    if not pending:
        print(f"Nothing to do. {len(existing)} tickets already in {RESULTS_PATH}", file=sys.stderr)
        return

    print(
        f"Running {len(pending)} ticket(s) ({len(existing)} already done) "
        f"with concurrency={args.concurrency}",
        file=sys.stderr,
    )

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(_run_one, run_ticket, ticket): ticket for ticket in pending}
        for future in as_completed(futures):
            ticket_id, record, exc = future.result()

            if exc is not None:
                log_call({"stage": "run_agent", "ticket_id": ticket_id, "event": "error", "error": str(exc)})
                print(f"[FAIL] {ticket_id}: {exc}", file=sys.stderr)
                failures.append(ticket_id)
                continue

            log_call({"stage": "run_agent", "ticket_id": ticket_id, "event": "success"})
            with _results_lock:
                existing[ticket_id] = record
                save_results(existing, dataset_order)  # persist after every ticket - resumable, no silent data loss
            print(f"[done] {ticket_id}", file=sys.stderr)

    print(f"Done. {len(existing)} tickets in {RESULTS_PATH}", file=sys.stderr)
    if failures:
        print(f"{len(failures)} ticket(s) failed after retries: {failures}. Re-run to retry them.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
