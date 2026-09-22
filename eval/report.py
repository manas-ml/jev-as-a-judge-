"""Build results/report.md and 3 PNG charts from repeatability_raw.csv + human_agreement.json.

Usage: python -m eval.report

Every number in the report is computed from the actual captured CSV/JSON produced by
repeatability.py and human_agreement.py - there are no placeholder numbers. If those
files don't exist yet (or are empty), this script says so and exits rather than
fabricating a report.
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval.common import DOES_PASS_THRESHOLD, ROOT, SEARCH_BEHAVIOR_SCORE

RAW_CSV = ROOT / "results" / "repeatability_raw.csv"
HUMAN_JSON = ROOT / "results" / "human_agreement.json"
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
    for (judge, ticket_id), group in df.groupby(["judge", "ticket_id"]):
        quality = group["quality"].dropna().to_numpy()
        variance = float(np.var(quality, ddof=1)) if len(quality) > 1 else float("nan")
        ci_lo, ci_hi = bootstrap_variance_ci(quality)
        rows.append(
            {
                "judge": judge,
                "ticket_id": ticket_id,
                "n": len(quality),
                "mean_quality": float(np.mean(quality)) if len(quality) else float("nan"),
                "variance": variance,
                "ci_95_lo": ci_lo,
                "ci_95_hi": ci_hi,
            }
        )
    variance_df = pd.DataFrame(rows)

    pivot = variance_df.pivot(index="ticket_id", columns="judge", values="variance")
    if "jev" in pivot.columns:
        for col in pivot.columns:
            if col == "jev":
                continue
            pivot[f"{col}_vs_jev_ratio"] = pivot[col] / pivot["jev"]
    variance_df = variance_df.merge(
        pivot.filter(like="_vs_jev_ratio").reset_index(), on="ticket_id", how="left"
    )

    bern_rows = []
    for (judge, ticket_id), group in df.groupby(["judge", "ticket_id"]):
        p = group["does_pass"].dropna().mean()
        bern_rows.append({"judge": judge, "ticket_id": ticket_id, "p_pass": p, "bernoulli_variance": p * (1 - p)})
    bernoulli_df = pd.DataFrame(bern_rows)

    return variance_df, bernoulli_df


def build_cost_latency_tables(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for judge, group in df.groupby("judge"):
        cost = group["cost"].dropna().to_numpy()
        latency = group["latency_s"].dropna().to_numpy()
        rows.append(
            {
                "judge": judge,
                "n_calls": len(group),
                "mean_cost_per_call": float(np.mean(cost)) if len(cost) else float("nan"),
                "cost_per_1000_evals": float(np.mean(cost) * 1000) if len(cost) else float("nan"),
                "latency_p50_s": float(np.percentile(latency, 50)) if len(latency) else float("nan"),
                "latency_p95_s": float(np.percentile(latency, 95)) if len(latency) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def make_charts(df: pd.DataFrame, variance_df: pd.DataFrame, cost_latency_df: pd.DataFrame) -> list[str]:
    charts = []

    # 1. Variance-ratio bar chart (per case, non-jev judges relative to jev)
    ratio_cols = [c for c in variance_df.columns if c.endswith("_vs_jev_ratio")]
    if ratio_cols:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plot_df = variance_df.drop_duplicates("ticket_id").set_index("ticket_id")[ratio_cols]
        plot_df.plot(kind="bar", ax=ax)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel("Variance ratio (judge / jev)")
        ax.set_title("Quality-score variance relative to Jev, per frozen case")
        fig.tight_layout()
        path = ROOT / "results" / "variance_ratio.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        charts.append(str(path))

    # 2. Quality-score distribution across repetitions (per judge)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for judge, group in df.groupby("judge"):
        ax.hist(group["quality"].dropna(), bins=20, alpha=0.5, label=judge)
    ax.set_xlabel("Quality score")
    ax.set_ylabel("Count across all repetitions/cases")
    ax.set_title("Quality-score distribution across repetitions")
    ax.legend()
    fig.tight_layout()
    path = ROOT / "results" / "quality_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    charts.append(str(path))

    # 3. Cost-vs-latency scatter (per call, colored by judge)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for judge, group in df.groupby("judge"):
        ax.scatter(group["latency_s"], group["cost"], alpha=0.5, label=judge, s=15)
    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("Cost per call ($)")
    ax.set_title("Cost vs. latency, per judge call")
    ax.legend()
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
    cost_latency_df = build_cost_latency_tables(df)
    chart_paths = make_charts(df, variance_df, cost_latency_df)

    human_agreement = {}
    if HUMAN_JSON.exists():
        human_agreement = json.loads(HUMAN_JSON.read_text())
    else:
        print(f"[warn] {HUMAN_JSON} not found - run `python -m eval.human_agreement` for the correctness numbers.", file=sys.stderr)

    n_reps = int(df.groupby(["judge", "ticket_id"]).size().max())
    n_cases = df["ticket_id"].nunique()

    lines = []
    lines.append("# Jev vs. Claude-as-Judge: Support-Ticket Agent Eval Report\n")
    lines.append(
        f"Generated from {len(df)} judge calls across {n_cases} frozen case(s), "
        f"up to {n_reps} repetition(s) per (judge, case) pair.\n"
    )
    lines.append(
        f"`search_behavior` -> quality mapping: {SEARCH_BEHAVIOR_SCORE}. "
        f"`does_pass` threshold: quality >= {DOES_PASS_THRESHOLD}.\n"
    )

    lines.append("## Repeatability: quality-score variance per case per judge\n")
    base_cols = [c for c in variance_df.columns if not c.endswith("_vs_jev_ratio")]
    lines.append(df_to_md(variance_df[base_cols]))
    lines.append("")
    ratio_cols = [c for c in variance_df.columns if c.endswith("_vs_jev_ratio")]
    if ratio_cols:
        lines.append("### Variance ratio vs. Jev (>1 means that judge is *less* repeatable than Jev on that case)\n")
        lines.append(df_to_md(variance_df.drop_duplicates("ticket_id")[["ticket_id"] + ratio_cols]))
        lines.append("")

    lines.append("## Binary variance: Bernoulli p(1-p) of does_pass, per case per judge\n")
    lines.append(df_to_md(bernoulli_df))
    lines.append("")

    lines.append("## Cost and latency per judge\n")
    lines.append(df_to_md(cost_latency_df))
    lines.append("")

    lines.append("## Human agreement (correctness signal)\n")
    if human_agreement:
        rows = []
        for judge, stats in human_agreement.items():
            rows.append(
                {
                    "judge": judge,
                    "n_labeled": stats["n"],
                    "cohen_kappa_does_pass": stats["cohen_kappa_does_pass"],
                    "spearman_rho_quality": stats["spearman_rho_quality"],
                    "spearman_p_value": stats["spearman_p_value"],
                    "pearson_r_quality": stats["pearson_r_quality"],
                    "pearson_p_value": stats["pearson_p_value"],
                }
            )
        lines.append(df_to_md(pd.DataFrame(rows)))
        lines.append("")
        for judge, stats in human_agreement.items():
            cm = stats["confusion_matrix_does_pass"]
            lines.append(f"**{judge} confusion matrix** (rows=human does_pass, cols=judge does_pass, labels {cm['labels']}):\n")
            lines.append("```")
            for row in cm["matrix"]:
                lines.append(str(row))
            lines.append("```\n")
    else:
        lines.append("_Not available - run `python -m eval.human_agreement`._\n")

    lines.append("## Charts\n")
    for path in chart_paths:
        filename = path.split("/")[-1]
        lines.append(f"![{filename}]({filename})\n")

    lines.append("## What this means\n")
    lines.append(
        "- **Small sample size.** This run covers "
        f"{n_cases} frozen case(s) repeated up to {n_reps}x for repeatability, and "
        f"{human_agreement.get(next(iter(human_agreement), ''), {}).get('n', 'a handful of')} labeled tickets "
        "for human agreement. These numbers are directional, not statistically definitive - treat any single-digit "
        "percentage-point difference between judges as noise until the sample is scaled up (N=100, more labeled tickets).\n"
    )
    lines.append(
        "- **One call per judge per ticket-evaluation, for both judges.** This is a deliberate deviation from the "
        "source repo (danielgshea/jev-as-a-judge), which used 3 Claude calls per case - most of why that benchmark's "
        "cost landed near $28. Because both judges here use a single-call design, the cost numbers in this report are "
        "*not* directly comparable to that figure; they mainly compare Jev's one-call-does-everything design against "
        "an equivalent one-call Claude-as-judge design, not against the original 3-call baseline.\n"
    )
    lines.append(
        "- **Repeatability is not correctness.** Low variance only means a judge gives a consistent score on the same "
        "input - a judge can be perfectly consistent and consistently wrong. The human-agreement numbers above "
        "(Cohen's kappa on does_pass, Spearman/Pearson on quality) are the actual correctness signal in this report; "
        "read the variance tables and the agreement table together, not the variance tables alone.\n"
    )

    REPORT_MD.write_text("\n".join(lines))
    print(f"Done. Report at {REPORT_MD}, charts: {chart_paths}", file=sys.stderr)


if __name__ == "__main__":
    main()
