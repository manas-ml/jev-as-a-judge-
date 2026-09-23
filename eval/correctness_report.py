"""Build results/correctness_report.md from human_agreement.json + human_agreement_calls.json.

Usage: python -m eval.correctness_report

Reads the aggregate agreement stats eval.human_agreement wrote, plus its per-ticket
cache, joins in each ticket's human_label from eval/dataset.json, and writes one
readable Markdown report. This is a separate report from results/report.md
(repeatability/cost/latency) - report.md answers "is Jev consistent?"; this one
answers "is Jev *right*?" No API calls; refuses to run if human_agreement.py's output
doesn't exist yet, rather than fabricating numbers.
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from eval.common import ROOT

DATASET_PATH = ROOT / "eval" / "dataset.json"
AGREEMENT_PATH = ROOT / "results" / "human_agreement.json"
CALLS_CACHE_PATH = ROOT / "results" / "human_agreement_calls.json"
REPORT_MD = ROOT / "results" / "correctness_report.md"


def df_to_md(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda v: "" if pd.isna(v) else float_fmt.format(v))
    return formatted.to_markdown(index=False)


def main() -> None:
    if not AGREEMENT_PATH.exists() or not CALLS_CACHE_PATH.exists():
        print(
            f"{AGREEMENT_PATH} and/or {CALLS_CACHE_PATH} not found. Run `python -m eval.human_agreement` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    agreement = json.loads(AGREEMENT_PATH.read_text())
    calls = json.loads(CALLS_CACHE_PATH.read_text())
    dataset = {t["id"]: t for t in json.load(open(DATASET_PATH))}

    if not agreement.get("ticket_ids"):
        print(f"{AGREEMENT_PATH} has no scored tickets yet. Run `python -m eval.human_agreement` first.", file=sys.stderr)
        sys.exit(1)

    rows = []
    for ticket_id in agreement["ticket_ids"]:
        human = dataset[ticket_id]["human_label"]
        jev = calls[ticket_id]
        rows.append(
            {
                "ticket_id": ticket_id,
                "human_does_pass": human["does_pass"],
                "jev_does_pass": jev["does_pass"],
                "match": "yes" if human["does_pass"] == jev["does_pass"] else "NO",
                "human_quality": human["quality"],
                "jev_quality": jev["quality"],
                "quality_delta": abs(human["quality"] - jev["quality"]),
            }
        )
    per_ticket_df = pd.DataFrame(rows).sort_values("quality_delta", ascending=False)
    n_mismatch = sum(1 for r in rows if r["match"] == "NO")

    lines = []
    lines.append("# Jev-as-Judge: Correctness vs. Human Labels\n")
    lines.append(
        f"Generated from {agreement['n']} labeled ticket(s) in `eval/dataset.json`, each scored once by Jev "
        "and compared against a hand-written `human_label`. This answers a different question from "
        "`report.md`: not \"is Jev consistent?\" but \"is Jev *right*?\"\n"
    )

    lines.append("## Agreement summary\n")
    summary_row = {
        "n_labeled": agreement["n"],
        "cohen_kappa_does_pass": agreement["cohen_kappa_does_pass"],
        "spearman_rho_quality": agreement["spearman_rho_quality"],
        "spearman_p_value": agreement["spearman_p_value"],
        "pearson_r_quality": agreement["pearson_r_quality"],
        "pearson_p_value": agreement["pearson_p_value"],
    }
    lines.append(df_to_md(pd.DataFrame([summary_row])))
    lines.append("")

    cm = agreement["confusion_matrix_does_pass"]
    lines.append(f"**Confusion matrix** (rows=human does_pass, cols=Jev does_pass, labels {cm['labels']}):\n")
    lines.append("```")
    for row in cm["matrix"]:
        lines.append(str(row))
    lines.append("```\n")

    lines.append(f"## Per-ticket comparison ({n_mismatch} does_pass mismatch(es), sorted by quality delta)\n")
    lines.append(df_to_md(per_ticket_df))
    lines.append("")

    lines.append("## What this means\n")
    lines.append(
        "- **Cohen's kappa** measures does_pass agreement beyond chance: <0 no agreement, 0-0.2 slight, "
        "0.2-0.4 fair, 0.4-0.6 moderate, 0.6-0.8 substantial, 0.8-1.0 near-perfect. It's null above when "
        "either the human labels or Jev's verdicts have no variance to measure agreement on.\n"
    )
    lines.append(
        "- **Spearman/Pearson on quality** measure how well Jev's continuous quality score tracks the "
        "human's, independent of the does_pass threshold - a high correlation with a low kappa would mean "
        "Jev ranks tickets similarly to a human but draws the pass/fail line in a different place.\n"
    )
    lines.append(
        f"- **Small sample size.** {agreement['n']} labeled tickets is enough to spot gross disagreement, "
        "not to certify Jev statistically - treat this as a smoke test for correctness, not a benchmark.\n"
    )
    lines.append(
        "- **One call per ticket, not repeated.** Unlike `report.md`'s repeatability sweep, each ticket here "
        "is scored once, so an unlucky single sample isn't averaged out. `human_agreement.py` caches "
        f"per-ticket results in `{CALLS_CACHE_PATH.name}` and won't re-score a cached ticket - delete its "
        "entry from that file to force a re-score.\n"
    )

    REPORT_MD.write_text("\n".join(lines))
    print(f"Done. Report at {REPORT_MD}", file=sys.stderr)


if __name__ == "__main__":
    main()
