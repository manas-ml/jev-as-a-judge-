"""Build results/report.md and 2 PNG charts from repeatability_raw.csv.

Usage: python -m eval.report

Every number in the report is computed from the actual captured CSV produced by
repeatability.py - there are no placeholder numbers. If that file doesn't exist yet
(or is empty), this script says so and exits rather than fabricating a report.
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval.common import DOES_PASS_THRESHOLD, ROOT, SEARCH_BEHAVIOR_SCORE

RAW_CSV = ROOT / "results" / "repeatability_raw.csv"
REPORT_MD = ROOT / "results" / "report.md"
RNG = np.random.default_rng(1234)  # fixed seed: report is reproducible given the same input CSV


def bootstrap_variance_ci(values: np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    boot_vars = np.empty(n_boot)
    n = len(values)
    for i in range(n_boot):
        sample = RNG.choice(values, size=n, replace=True)
        boot_vars[i] = np.var(sample, ddof=1)
    return (float(np.percentile(boot_vars, 2.5)), float(np.percentile(boot_vars, 97.5)))


def build_repeatability_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for ticket_id, group in df.groupby("ticket_id"):
        quality = group["quality"].dropna().to_numpy()
        variance = float(np.var(quality, ddof=1)) if len(quality) > 1 else float("nan")
        ci_lo, ci_hi = bootstrap_variance_ci(quality)
        rows.append(
            {
                "ticket_id": ticket_id,
                "n": len(quality),
                "mean_quality": float(np.mean(quality)) if len(quality) else float("nan"),
                "variance": variance,
                "ci_95_lo": ci_lo,
                "ci_95_hi": ci_hi,
            }
        )
    variance_df = pd.DataFrame(rows)

    bern_rows = []
    for ticket_id, group in df.groupby("ticket_id"):
        p = group["does_pass"].dropna().mean()
        bern_rows.append({"ticket_id": ticket_id, "p_pass": p, "bernoulli_variance": p * (1 - p)})
    bernoulli_df = pd.DataFrame(bern_rows)

    return variance_df, bernoulli_df


def build_cost_latency_table(df: pd.DataFrame) -> pd.DataFrame:
    cost = df["cost"].dropna().to_numpy()
    latency = df["latency_s"].dropna().to_numpy()
    return pd.DataFrame(
        [
            {
                "n_calls": len(df),
                "mean_cost_per_call": float(np.mean(cost)) if len(cost) else float("nan"),
                "cost_per_1000_evals": float(np.mean(cost) * 1000) if len(cost) else float("nan"),
                "latency_p50_s": float(np.percentile(latency, 50)) if len(latency) else float("nan"),
                "latency_p95_s": float(np.percentile(latency, 95)) if len(latency) else float("nan"),
            }
        ]
    )


def make_charts(df: pd.DataFrame) -> list[str]:
    charts = []

    # 1. Quality-score distribution across repetitions
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df["quality"].dropna(), bins=20, alpha=0.7)
    ax.set_xlabel("Quality score")
    ax.set_ylabel("Count across all repetitions/cases")
    ax.set_title("Jev quality-score distribution across repetitions")
    fig.tight_layout()
    path = ROOT / "results" / "quality_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    charts.append(str(path))

    # 2. Cost-vs-latency scatter (per call)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(df["latency_s"], df["cost"], alpha=0.6, s=15)
    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("Cost per call ($)")
    ax.set_title("Jev cost vs. latency, per call")
    fig.tight_layout()
    path = ROOT / "results" / "cost_vs_latency.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    charts.append(str(path))

    return charts


def df_to_md(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda v: "" if pd.isna(v) else float_fmt.format(v))
    return formatted.to_markdown(index=False)


def main() -> None:
    if not RAW_CSV.exists():
        print(f"{RAW_CSV} does not exist yet. Run `python -m eval.repeatability` first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(RAW_CSV)
    if df.empty:
        print(f"{RAW_CSV} is empty. Run `python -m eval.repeatability` first.", file=sys.stderr)
        sys.exit(1)

    variance_df, bernoulli_df = build_repeatability_tables(df)
    cost_latency_df = build_cost_latency_table(df)
    chart_paths = make_charts(df)

    n_reps = int(df.groupby("ticket_id").size().max())
    n_cases = df["ticket_id"].nunique()

    lines = []
    lines.append("# Jev-as-Judge: Support-Ticket Agent Eval Report\n")
    lines.append(
        f"Generated from {len(df)} Jev judge calls across {n_cases} frozen case(s), "
        f"up to {n_reps} repetition(s) per case.\n"
    )
    lines.append(
        f"`search_behavior` -> quality mapping: {SEARCH_BEHAVIOR_SCORE}. "
        f"`does_pass` threshold: quality >= {DOES_PASS_THRESHOLD}.\n"
    )

    lines.append("## Repeatability: quality-score variance per case\n")
    lines.append(df_to_md(variance_df))
    lines.append("")

    lines.append("## Binary variance: Bernoulli p(1-p) of does_pass, per case\n")
    lines.append(df_to_md(bernoulli_df))
    lines.append("")

    lines.append("## Cost and latency\n")
    lines.append(df_to_md(cost_latency_df))
    lines.append("")

    lines.append("## Charts\n")
    for path in chart_paths:
        filename = path.split("/")[-1]
        lines.append(f"![{filename}]({filename})\n")

    lines.append("## What this means\n")
    scale_up_note = "" if n_reps >= 100 else " until the sample is scaled up (N=100)"
    lines.append(
        "- **Small sample size.** This run covers "
        f"{n_cases} frozen case(s) repeated up to {n_reps}x for repeatability. These numbers are directional, "
        f"not statistically definitive - treat any single-digit percentage-point movement as noise{scale_up_note}.\n"
    )
    lines.append(
        "- **One call per ticket-evaluation.** Jev's `state` carries all evidence and answers all three typed "
        "questions (`groundedness`, `search_behavior`, `usefulness`) in a single `system_one.create()` call.\n"
    )
    lines.append(
        "- **Repeatability is not correctness.** Low variance only means Jev gives a consistent score on the same "
        "input - a judge can be perfectly consistent and consistently wrong. This report does not measure "
        "agreement with human judgment; it only measures how stable Jev's own scores are across repeated calls.\n"
    )

    REPORT_MD.write_text("\n".join(lines))
    print(f"Done. Report at {REPORT_MD}, charts: {chart_paths}", file=sys.stderr)


if __name__ == "__main__":
    main()
