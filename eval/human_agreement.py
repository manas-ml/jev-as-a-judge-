"""Compare the Jev judge's verdict against hand-written human labels.

Usage:
  python -m eval.human_agreement
  python -m eval.human_agreement --concurrency 10

For every ticket in eval/dataset.json that has a human_label, runs the Jev judge once
against the frozen agent answer in eval/results.json, then computes:
  - Cohen's kappa on does_pass (binary agreement)
  - Spearman rho and Pearson r (with p-values) on quality
  - a confusion matrix

This is the number that answers "is the low variance actually correct?" - repeatability
alone only tells you Jev is consistent, not that it's consistent with the truth.

Resumable: per-ticket judge calls are cached in results/human_agreement_calls.json so a
failure partway through doesn't force re-spending on tickets already judged. Calls run
concurrently (bounded by --concurrency, default 10) via a thread pool, same pattern as
repeatability.py.

Run `python -m eval.correctness_report` afterward to build a readable Markdown summary
from what this script writes - that's a separate report from results/report.md, which
covers repeatability/cost/latency, not correctness.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from eval.common import ROOT, compute_quality, load_env, log_call

DATASET_PATH = ROOT / "eval" / "dataset.json"
RESULTS_PATH = ROOT / "eval" / "results.json"
CALLS_CACHE_PATH = ROOT / "results" / "human_agreement_calls.json"
OUT_PATH = ROOT / "results" / "human_agreement.json"

_cache_lock = threading.Lock()


def load_labeled_cases() -> list[dict]:
    dataset = {t["id"]: t for t in json.load(open(DATASET_PATH))}
    results = {r["id"]: r for r in json.load(open(RESULTS_PATH))}
    cases = []
    for ticket_id, ticket in dataset.items():
        if not ticket.get("human_label"):
            continue
        if ticket_id not in results:
            print(f"[warn] {ticket_id} has a human_label but no entry in results.json - skipping", file=sys.stderr)
            continue
        case = dict(results[ticket_id])
        case["human_label"] = ticket["human_label"]
        cases.append(case)
    return cases


def load_cache() -> dict:
    if CALLS_CACHE_PATH.exists():
        return json.load(open(CALLS_CACHE_PATH))
    return {}


def save_cache(cache: dict) -> None:
    CALLS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALLS_CACHE_PATH.write_text(json.dumps(cache, indent=2, default=str))


def _run_one(jev_judge, case: dict) -> tuple[str, dict | None, Exception | None]:
    try:
        result = jev_judge.evaluate(case)
    except Exception as exc:  # noqa: BLE001
        return case["id"], None, exc
    quality = compute_quality(result["groundedness"], result["search_behavior"], result["usefulness"])
    entry = {
        "does_pass": result.get("does_pass"),
        "quality": quality,
        "groundedness": result["groundedness"],
        "search_behavior": result["search_behavior"],
        "usefulness": result["usefulness"],
    }
    return case["id"], entry, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent Jev calls in flight")
    args = parser.parse_args()

    load_env()
    from eval.judges import jev_judge

    cases = load_labeled_cases()
    cache = load_cache()

    pending = [c for c in cases if c["id"] not in cache]
    if pending:
        print(
            f"Running {len(pending)} ticket(s) ({len(cases) - len(pending)} cached) "
            f"with concurrency={args.concurrency}",
            file=sys.stderr,
        )
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(_run_one, jev_judge, case) for case in pending]
            for future in as_completed(futures):
                ticket_id, entry, exc = future.result()
                if exc is not None:
                    log_call({"stage": "human_agreement", "ticket_id": ticket_id, "event": "error", "error": str(exc)})
                    print(f"  FAILED: {ticket_id}: {exc} - will retry on next invocation", file=sys.stderr)
                    continue
                with _cache_lock:
                    cache[ticket_id] = entry
                    save_cache(cache)
                print(f"[done] {ticket_id}", file=sys.stderr)
    else:
        print(f"Nothing to do - all {len(cases)} labeled ticket(s) already cached.", file=sys.stderr)

    ids = [c["id"] for c in cases if c["id"] in cache]
    if len(ids) < len(cases):
        missing = set(c["id"] for c in cases) - set(ids)
        print(f"[warn] missing results for {missing}; excluding from stats", file=sys.stderr)

    human_pass = np.array([cases_by_id(cases, i)["human_label"]["does_pass"] for i in ids])
    judge_pass = np.array([cache[i]["does_pass"] for i in ids])
    human_quality = np.array([cases_by_id(cases, i)["human_label"]["quality"] for i in ids])
    judge_quality = np.array([cache[i]["quality"] for i in ids])

    if len(set(human_pass)) > 1 or len(set(judge_pass)) > 1:
        kappa = cohen_kappa_score(human_pass, judge_pass)
        kappa = None if np.isnan(kappa) else float(kappa)
    else:
        kappa = None  # undefined: no variance in labels/predictions to measure agreement on
    spearman = stats.spearmanr(human_quality, judge_quality)
    pearson = stats.pearsonr(human_quality, judge_quality)
    cm = confusion_matrix(human_pass, judge_pass, labels=[0, 1]).tolist()

    report = {
        "n": len(ids),
        "cohen_kappa_does_pass": kappa,
        "spearman_rho_quality": _clean(spearman.correlation),
        "spearman_p_value": _clean(spearman.pvalue),
        "pearson_r_quality": _clean(pearson.statistic),
        "pearson_p_value": _clean(pearson.pvalue),
        "confusion_matrix_does_pass": {"labels": [0, 1], "matrix": cm},
        "ticket_ids": ids,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Done. Aggregate stats in {OUT_PATH}. Run `python -m eval.correctness_report` for a readable summary.", file=sys.stderr)


def _clean(value: float) -> float | None:
    value = float(value)
    return None if np.isnan(value) else value


def cases_by_id(cases: list[dict], ticket_id: str) -> dict:
    for c in cases:
        if c["id"] == ticket_id:
            return c
    raise KeyError(ticket_id)


if __name__ == "__main__":
    main()
