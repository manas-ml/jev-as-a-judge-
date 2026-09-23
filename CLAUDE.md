# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Python PoC demonstrating **Jev** (TypeSafe's structured decision model, via
OpenRouter's System One API) as an LLM-as-judge pattern for scoring a support-ticket
resolution agent. It measures repeatability (score variance on frozen, repeated
inputs), cost, latency (rolled into `results/report.md`), and separately, agreement
with hand-written human labels - i.e. is Jev not just consistent, but *correct*
(rolled into its own `results/correctness_report.md`).

The agent and the judge both talk to models through **OpenRouter** —
`OPENROUTER_API_KEY` is the one credential needed, and `AGENT_MODEL_ID` /
`JEV_MODEL_ID` in `.env` select which model each role uses.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENROUTER_API_KEY
```

All commands run from the repo root using `python -m` so the `agent`/`eval` packages resolve.

## Pipeline commands (run in this order)

1. `python -m eval.judges.jev_judge` — one live smoke-test call; prints parsed JSON or
   exits non-zero with the raw response on stderr (use this to debug parsing when
   OpenRouter's shape drifts).
2. `python -m eval.run_agent --concurrency 5` — runs the Strands agent over every
   ticket in `eval/dataset.json`, writes `eval/results.json`. Tickets run concurrently
   via a thread pool (bounded by `--concurrency`, default 5); each ticket gets its own
   fresh `Agent` instance so there's no shared mutable state across threads. Resumable
   (skips tickets already present); `--force` regenerates everything. A ticket that
   fails after retries is logged and skipped (not a hard abort) — re-run to retry it.
   **Never re-run between judge runs** — that reintroduces agent-side variance into
   what's supposed to be judge-only variance.
3. `python -m eval.repeatability --n 20` (smoke) then `--n 100 --concurrency 20`
   (real run) — replays the 5 frozen cases through Jev N times, appends to
   `results/repeatability_raw.csv`. Calls run concurrently via a thread pool (bounded
   by `--concurrency`, default 10) — this only cuts wall-clock time, not API call count
   or cost, since System One's `create()` takes one ticket per request with no batch
   parameter. Resumable per (ticket, rep).
4. `python -m eval.report` — computes `results/report.md` + 2 PNGs
   (`quality_distribution.png`, `cost_vs_latency.png`) purely from the CSV above.
   No placeholder numbers: if the input file doesn't exist, it says so and exits rather
   than fabricating output. Answers "is Jev consistent?", not "is Jev right?".
5. `python -m eval.human_agreement --concurrency 10` — scores all 25 labeled tickets
   against `eval/dataset.json`'s `human_label`, once each (not repeated), concurrently
   via a thread pool. Writes `results/human_agreement.json` (Cohen's kappa on
   `does_pass`, Spearman/Pearson on `quality`, confusion matrix). Resumable — caches
   per-ticket results in `results/human_agreement_calls.json`.
6. `python -m eval.correctness_report` — reads `human_agreement.json` +
   `human_agreement_calls.json`, joins in each ticket's `human_label`, and writes
   `results/correctness_report.md`: the aggregate stats plus a per-ticket
   human-vs-Jev table sorted by disagreement. Separate report from `report.md` — this
   one answers "is Jev right?".

Every API call (agent + judge, including retries) is appended to `results/calls.jsonl`
(gitignored) for auditing.

## Architecture

**`agent/ticket_agent.py`** — the thing being evaluated. A Strands `Agent` with one
tool, `kb_search`, over a fixed 15-entry JSON KB (`agent/kb.json`). Retrieval is
deterministic TF-IDF cosine similarity with Porter stemming (not an LLM embedding call)
so repeated runs on the same ticket are reproducible — variance measured downstream
comes from the judge, not the retrieval step. `run_ticket()` builds a fresh `Agent`
per call (no cross-ticket history) and returns a frozen record: `answer`, `tool_calls`
(the queries sent to `kb_search`), `kb_evidence` (deduped KB entries actually returned).
This module knows nothing about judging or evaluation.

**`eval/judges/jev_judge.py`** — the judge. Implements `evaluate(case, model_id=None)
-> dict`, scoring three axes in a single `system_one.create()` call: `groundedness`
(0-1), `search_behavior` (`appropriate|unnecessary|failed`), `usefulness` (0-1). Parses
the SDK's response defensively (never assumes a field beyond what the SDK's own schema
guarantees) since this is a live third-party API.

**`eval/common.py`** — shared plumbing everything above depends on: `.env` loading,
`retry_with_backoff` (exponential + jitter on 429/5xx and connection errors, used by
every agent/judge call), `log_call` (the JSONL audit log), and the composite-score
math. `compute_quality()` folds the three-axis judgment into one number via
`SEARCH_BEHAVIOR_SCORE` (`appropriate=1.0, unnecessary=0.5, failed=0.0`) —
`quality = mean(groundedness, mapped_search_behavior, usefulness)`.
`derive_does_pass()` applies `DOES_PASS_THRESHOLD = 0.5` to that composite score.

**`eval/dataset.json`** — 25 tickets, 5 marked `frozen: true` (what `repeatability.py`
replays), all 25 carrying a `human_label` (`does_pass`, `quality`, `notes`) used by
`human_agreement.py`. Includes three deliberate trap cases testing over/under-searching
(`t17`–`t19`: KB genuinely doesn't cover the topic, correct behavior is honest
escalation not fabrication; `t20`: a pure thank-you that should NOT trigger
`kb_search` at all).

**`eval/human_agreement.py`** — the correctness check. Runs `jev_judge.evaluate()`
once per labeled ticket (not repeated), concurrently, and computes Cohen's kappa
(`does_pass` agreement) plus Spearman/Pearson (`quality` correlation) against each
ticket's `human_label`. Writes aggregate stats to `results/human_agreement.json` and
caches every per-ticket score in `results/human_agreement_calls.json` (resumable —
a ticket already in the cache is never re-scored).

**`eval/correctness_report.py`** — pure read of `human_agreement.json` +
`human_agreement_calls.json` (no API calls), joined against `dataset.json`'s
`human_label`. Writes `results/correctness_report.md`: the aggregate stats plus a
per-ticket table sorted by disagreement, so a mismatch is traceable to a specific
ticket, not just a summary number.

**Data flow**: `dataset.json` → `run_agent.py` → `results.json` (frozen agent output,
judge-input) → two independent consumers, both calling `eval/judges/jev_judge.evaluate()`
against `results.json` but never each other: `repeatability.py` (N reps × 5 frozen
tickets → `repeatability_raw.csv` → `report.py` → `report.md`, "is Jev consistent?")
and `human_agreement.py` (1 rep × 25 labeled tickets → `human_agreement.json` →
`correctness_report.py` → `correctness_report.md`, "is Jev right?").

## Working in this repo

- The agent and judge are two independent `evaluate`/`run_ticket` call sites — when
  changing the scoring contract (the three axes, the schema), update `jev_judge.py`
  *and* `eval/common.py`'s `compute_quality`/`derive_does_pass` together, since
  `repeatability.py` and `human_agreement.py` both assume they stay in lockstep.
- Every stage that costs money (`run_agent`, `repeatability`, `human_agreement`) is
  resumable by design (skip-if-already-present + persist-after-every-item) — preserve
  that property in any change to these scripts, since a real sweep is a paid, long-running
  operation that will get interrupted.
- `results/` (except the two `*_report.md` files' prerequisites) and `eval/results.json`
  are real run output, not fixtures — don't hand-edit or fabricate values in them;
  `report.py` and `correctness_report.py` are both deliberately built to refuse to run
  rather than produce placeholder numbers.
- `report.py` and `correctness_report.py` are two separate reports on purpose —
  repeatability/cost/latency vs. correctness are different questions with different
  data sources (`repeatability_raw.csv` vs. `human_agreement.json`), and conflating
  them into one report was the earlier design this build moved away from. Keep them
  separate rather than merging.
- This PoC previously compared Jev against a second OpenRouter-hosted judge
  (DeepSeek); that comparison path was removed so the demo shows only the
  Jev-as-judge pattern end to end. If it's reintroduced later, keep the
  `evaluate(case, model_id=None) -> dict` contract so it can reuse both
  `repeatability.py` and `human_agreement.py` with minimal changes.
