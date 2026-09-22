"""Replay each frozen ticket N times through both judges to measure score variance.

Usage:
  python -m eval.repeatability --n 20     # cheap smoke test
  python -m eval.repeatability --n 100    # real run

Persists every single repetition's raw output + cost + latency to
results/repeatability_raw.csv (append-only). Resumable: reps already recorded for a
given (judge, ticket_id, rep) are skipped on a re-run, so a partial N=100 sweep that
died partway through can just be re-invoked with the same --n.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from eval.common import ROOT, compute_quality, load_env, log_call

RESULTS_PATH = ROOT / "eval" / "results.json"
OUT_PATH = ROOT / "results" / "repeatability_raw.csv"

FIELDNAMES = [
    "judge",
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


def load_done_reps() -> set[tuple[str, str, int]]:
    if not OUT_PATH.exists():
        return set()
    done = set()
    with open(OUT_PATH) as f:
        for row in csv.DictReader(f):
            done.add((row["judge"], row["ticket_id"], int(row["rep"])))
    return done


def append_row(row: dict) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not OUT_PATH.exists()
    with open(OUT_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20, help="Repetitions per case per judge")
    args = parser.parse_args()

    load_env()
    from eval.judges import claude_judge, jev_judge

    judges = {"jev": jev_judge.evaluate, "claude": claude_judge.evaluate}

    cases = load_frozen_cases()
    done = load_done_reps()

    total = len(cases) * len(judges) * args.n
    completed = 0

    for case in cases:
        for judge_name, judge_fn in judges.items():
            for rep in range(1, args.n + 1):
                completed += 1
                key = (judge_name, case["id"], rep)
                if key in done:
                    continue
                print(f"[{completed}/{total}] {judge_name} {case['id']} rep {rep}", file=sys.stderr)
                try:
                    result = judge_fn(case)
                except Exception as exc:  # noqa: BLE001
                    log_call(
                        {
                            "stage": "repeatability",
                            "judge": judge_name,
                            "ticket_id": case["id"],
                            "rep": rep,
                            "event": "error",
                            "error": str(exc),
                        }
                    )
                    print(f"  FAILED after retries: {exc}", file=sys.stderr)
                    # Don't record a row for this rep - a re-run of the same --n will retry it.
                    continue

                # Quality is always recomputed here (rather than trusting each judge's own
                # 'quality' key, which only jev_judge populates) so both judges' composite
                # score comes from the exact same formula and is directly comparable.
                quality = compute_quality(result["groundedness"], result["search_behavior"], result["usefulness"])
                append_row(
                    {
                        "judge": judge_name,
                        "ticket_id": case["id"],
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

    print(f"Done. Raw repetitions in {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
