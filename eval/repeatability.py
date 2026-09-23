"""Replay each frozen ticket N times through the Jev judge to measure score variance.

Usage:
  python -m eval.repeatability --n 20                    # cheap smoke test
  python -m eval.repeatability --n 100                    # the real run
  python -m eval.repeatability --n 100 --concurrency 20   # more calls in flight

Persists every single repetition's raw output + cost + latency to
results/repeatability_raw.csv (append-only). Resumable: reps already recorded for a
given (ticket_id, rep) are skipped on a re-run, so a partial N=100 sweep that died
partway through can just be re-invoked with the same --n.

Calls run concurrently, bounded by --concurrency (default 10), via a thread pool -
jev_judge.evaluate() is a blocking HTTP call, so threads parallelize wall-clock time
without touching the judge's retry/backoff logic. This does not reduce the number of
API calls or the cost of a sweep - System One's create() endpoint takes one `state`
per request with no batch parameter - it only cuts how long the sweep takes to run.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from eval.common import ROOT, compute_quality, load_env, log_call

RESULTS_PATH = ROOT / "eval" / "results.json"
OUT_PATH = ROOT / "results" / "repeatability_raw.csv"

FIELDNAMES = [
    "ticket_id",
    "rep",
    "groundedness",
    "search_behavior",
    "usefulness",
    "quality",
    "does_pass",
    "input_tokens",
    "output_tokens",
    "cost",
    "latency_s",
]

_csv_lock = threading.Lock()


def load_frozen_cases() -> list[dict]:
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    frozen_ids = {t["id"] for t in json.load(open(ROOT / "eval" / "dataset.json")) if t.get("frozen")}
    cases = [r for r in results if r["id"] in frozen_ids]
    missing = frozen_ids - {c["id"] for c in cases}
    if missing:
        raise RuntimeError(
            f"Frozen ticket(s) {missing} not found in {RESULTS_PATH}. Run `python -m eval.run_agent` first."
        )
    return cases


def load_done_reps() -> set[tuple[str, int]]:
    if not OUT_PATH.exists():
        return set()
    done = set()
    with open(OUT_PATH) as f:
        for row in csv.DictReader(f):
            done.add((row["ticket_id"], int(row["rep"])))
    return done


def append_row(row: dict) -> None:
    # Multiple worker threads call this concurrently - serialize the read-header/append
    # so rows from different threads never interleave mid-line in the CSV.
    with _csv_lock:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        is_new = not OUT_PATH.exists()
        with open(OUT_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if is_new:
                writer.writeheader()
            writer.writerow(row)


def _run_one(jev_judge, case: dict, rep: int) -> tuple[str, int, dict | None, Exception | None]:
    try:
        result = jev_judge.evaluate(case)
    except Exception as exc:  # noqa: BLE001
        return case["id"], rep, None, exc
    return case["id"], rep, result, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20, help="Repetitions per frozen case")
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent Jev calls in flight")
    args = parser.parse_args()

    load_env()
    from eval.judges import jev_judge

    cases = load_frozen_cases()
    done = load_done_reps()

    total = len(cases) * args.n
    work = [(case, rep) for case in cases for rep in range(1, args.n + 1) if (case["id"], rep) not in done]

    if not work:
        print(f"Nothing to do - all {total} (ticket, rep) combinations already recorded.", file=sys.stderr)
        return

    print(
        f"Running {len(work)} calls ({total - len(work)} already recorded) "
        f"with concurrency={args.concurrency}",
        file=sys.stderr,
    )

    completed = total - len(work)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(_run_one, jev_judge, case, rep) for case, rep in work]
        for future in as_completed(futures):
            ticket_id, rep, result, exc = future.result()
            completed += 1

            if exc is not None:
                log_call(
                    {
                        "stage": "repeatability",
                        "ticket_id": ticket_id,
                        "rep": rep,
                        "event": "error",
                        "error": str(exc),
                    }
                )
                print(f"[{completed}/{total}] {ticket_id} rep {rep} FAILED after retries: {exc}", file=sys.stderr)
                # Don't record a row for this rep - a re-run of the same --n will retry it.
                continue

            # Quality is always recomputed here (rather than trusting jev_judge's own
            # 'quality' key) so the composite score always comes from the exact same
            # formula in eval/common.py.
            quality = compute_quality(result["groundedness"], result["search_behavior"], result["usefulness"])
            append_row(
                {
                    "ticket_id": ticket_id,
                    "rep": rep,
                    "groundedness": result["groundedness"],
                    "search_behavior": result["search_behavior"],
                    "usefulness": result["usefulness"],
                    "quality": quality,
                    "does_pass": result.get("does_pass"),
                    "input_tokens": result.get("input_tokens"),
                    "output_tokens": result.get("output_tokens"),
                    "cost": result.get("cost"),
                    "latency_s": result.get("latency_s"),
                }
            )
            print(f"[{completed}/{total}] {ticket_id} rep {rep} done", file=sys.stderr)

    print(f"Done. Raw repetitions in {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
