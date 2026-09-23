# Architecture

This PoC has two independent pieces talking to two independent OpenRouter-hosted
models: the **support-ticket agent** being evaluated (`AGENT_MODEL_ID`), and the
**Jev judge** scoring its output (`JEV_MODEL_ID`). Everything else is a pipeline of
small scripts that turn one into frozen input for the other, then turn the judge's
verdicts into two separate reports answering two separate questions: is Jev
*consistent* (`report.md`), and is Jev *correct* (`correctness_report.md`).

## Component map

```
                              .env (OPENROUTER_API_KEY, AGENT_MODEL_ID, JEV_MODEL_ID)
                                            |
                                            v
  +----------------------+        +------------------------+
  | agent/ticket_agent.py |        | eval/judges/jev_judge.py |
  |------------------------|        |------------------------|
  | Strands Agent          |        | evaluate(case)          |
  | + kb_search tool       |        | -> system_one.create()  |
  | (TF-IDF over kb.json)  |        | (OpenRouter System One) |
  +-----------+------------+        +-----------+------------+
              ^                                  ^      ^
              | run_ticket(question)             |      |
              |                                  | evaluate(case), N reps
              |                                  | per frozen ticket
  +-----------+------------+        +------------+-----------+
  | eval/run_agent.py       |        | eval/repeatability.py    |
  | (thread pool,            |        | (thread pool,             |
  |  --concurrency)          |------->|  --concurrency, --n)      |
  +-----------+--------------+ writes +------------+--------------+
              |                                     |
              v                                     v
     eval/results.json                results/repeatability_raw.csv
     (frozen agent answers)            (every rep: scores, cost, latency)
              |                                     |
              |                                     v
              |                          +----------------------+
              |                          | eval/report.py         |
              |                          +-----------+-------------+
              |                                      |
              |                                      v
              |                       results/report.md + quality_distribution.png
              |                                            + cost_vs_latency.png
              |                                    ("is Jev consistent?")
              |
              |    evaluate(case), 1x per labeled ticket
              v                                      ^
   +------------------------+                        |
   | eval/human_agreement.py |------------------------+
   | (thread pool,            |
   |  --concurrency)          |
   +-----------+--------------+ writes
               v
    results/human_agreement.json
    results/human_agreement_calls.json
               |
               v
    +---------------------------+
    | eval/correctness_report.py |
    +-------------+---------------+
                  |
                  v
       results/correctness_report.md
          ("is Jev right?" - vs. eval/dataset.json's human_label)

  eval/common.py underlies all of the above: .env loading, retry_with_backoff,
  log_call() -> results/calls.jsonl (every API call, gitignored), and the
  compute_quality()/derive_does_pass() scoring math shared by jev_judge.py,
  repeatability.py, and human_agreement.py.
```

## Execution walkthrough: a full run, start to finish

| Step | Command | What happens |
|---|---|---|
| 0. Setup | `cp .env.example .env` | Fill in `OPENROUTER_API_KEY`, `AGENT_MODEL_ID`, `JEV_MODEL_ID`. No code runs yet. |
| 1. Smoke test (optional) | `python -m eval.judges.jev_judge` | One live call against a hardcoded sample ticket, to catch response-shape drift before spending real money on a sweep. |
| 2. Freeze agent answers | `python -m eval.run_agent --concurrency 5` | Runs every dataset ticket through a fresh Agent (concurrently), which calls `kb_search` as needed and answers the ticket. Writes `eval/results.json`. |
| 3. Repeatability sweep | `python -m eval.repeatability --n 100 --concurrency 20` | Replays each of the 5 frozen tickets through Jev N times (concurrently), one API call per rep scoring all 3 axes at once. Appends `results/repeatability_raw.csv`. |
| 4. Build repeatability report | `python -m eval.report` | Reads the CSV, computes variance/cost/latency stats and two charts, writes `results/report.md`. No API calls. Answers "is Jev *consistent*?" |
| 5. Correctness check | `python -m eval.human_agreement --concurrency 10` | Scores each of the 25 labeled tickets through Jev once (concurrently), compares against `human_label`. Writes `results/human_agreement.json` + a per-ticket cache. |
| 6. Build correctness report | `python -m eval.correctness_report` | Reads the agreement stats + per-ticket cache, joins in `human_label`, writes `results/correctness_report.md`. No API calls. Answers "is Jev *right*?" |

Every step is resumable - each skips work already recorded by a prior run. Every API
call made along the way (agent or judge) is also logged to `results/calls.jsonl` as a
raw audit trail. Steps 3-4 and steps 5-6 are independent pairs - neither depends on
the other, only both depend on step 2's frozen `eval/results.json`.

## Data flow 1: freezing agent answers (`eval/run_agent.py`)

One HTTP round trip per ticket, to the **agent** model. Fans out across a thread
pool (`--concurrency`, default 5); each ticket gets its own fresh `Agent` instance,
so nothing is shared across threads except the read-only TF-IDF index built once at
import time.

```
eval/dataset.json                    agent/kb.json (15 fixed FAQ entries)
  25 tickets                                 |
     |                                       v
     |                          +-------------------------+
     |                          | TF-IDF vectorizer +      |
     |                          | Porter stemmer            |
     |                          | (built once at import)    |
     |                          +-------------------------+
     |                                       ^
     v                                       | kb_search(query) -- deterministic,
+---------------------+   build_agent()      | same query -> same result every time
| eval/run_agent.py    |------------------+   |
|  ThreadPoolExecutor   |                  v   |
|  (--concurrency)      |         +--------------------+
|                        |-------->| Strands Agent       |
|  for each ticket:      |  run_   | (agent/ticket_agent  |
|    retry_with_backoff( |  ticket | .py)                 |
|      run_ticket(q) )   |  (q)    |  uses kb_search tool  |
|                        |<--------|  as needed             |
+-----------+------------+         +----------+-----------+
            |                                 |
            | answer, tool_calls,             | OpenRouter chat-completions call
            | kb_evidence (deduped)           | to AGENT_MODEL_ID
            v                                 v
   eval/results.json                (results/calls.jsonl audit log,
   [{id, question, answer,           via eval.common.log_call)
    kb_evidence, tool_calls,
    expected_behavior}, ...]
   -- written in dataset order after
      EVERY ticket (resumable; a
      ticket that still fails after
      retries is logged+skipped,
      not a hard abort)
```

## Data flow 2: repeatability sweep + judging (`eval/repeatability.py`)

One HTTP round trip per (ticket, repetition), to the **Jev** model. `eval/results.json`
from flow 1 is frozen input here - `repeatability.py` never re-invokes the agent, only
the judge, so all variance measured downstream is judge-side, not agent-side.

```
eval/results.json                    eval/dataset.json
(frozen agent answers,                (frozen: true flags which
 keyed by ticket id)                   5 of the 25 tickets get replayed)
        |                                     |
        +------------------+------------------+
                            v
                load_frozen_cases()
                -- 5 frozen tickets only
                            |
                            v
              +--------------------------+
              | eval/repeatability.py     |
              | ThreadPoolExecutor         |
              | (--concurrency, default 10)|
              |                             |
              | work = every (ticket, rep)  |
              | in 1..N not already in the  |
              | CSV (resumable)             |
              +-------------+---------------+
                            |
              fan out N reps x 5 tickets, each:
                            |
                            v
              +--------------------------+
              | eval/judges/jev_judge.py  |
              | evaluate(case)             |
              |                             |
              |  state = {question, answer, |
              |   kb_evidence, expected_    |
              |   behavior}                 |
              |  questions = {groundedness, |
              |   search_behavior,          |
              |   usefulness}   -- ALL 3    |
              |   asked in ONE call         |
              |                             |
              |  client.system_one.create(  |
              |    model=JEV_MODEL_ID,       |
              |    questions=..., state=...) |
              +-------------+---------------+
                            |
                 OpenRouter System One API
                 (POST /api/v1/systemone)
                            |
                            v
              DecisionsResponse: answers{}, usage{}
                            |
              quality = compute_quality(          <- eval/common.py
                groundedness, search_behavior,        (identical formula every call)
                usefulness)
              does_pass = derive_does_pass(quality)
                            |
                            v
           append_row() -- lock-protected, thread-safe
                            |
                            v
           results/repeatability_raw.csv (append-only)
           one row per (ticket_id, rep): scores,
           does_pass, tokens, cost, latency_s
```

## Data flow 3: building the repeatability report (`eval/report.py`)

Pure read of `results/repeatability_raw.csv` - no API calls, no re-judging. Refuses
to run (rather than fabricate numbers) if the CSV doesn't exist yet or is empty.

```
results/repeatability_raw.csv
        |
        v
+---------------------------+
| eval/report.py              |
|                              |
| build_repeatability_tables() | -> per-ticket variance + 95% bootstrap CI,
|                              |    Bernoulli p(1-p) of does_pass
| build_cost_latency_table()   | -> mean cost/call, cost per 1000 evals,
|                              |    p50/p95 latency
| make_charts()                | -> quality_distribution.png (histogram)
|                              |    cost_vs_latency.png (scatter)
+---------------+-------------+
                |
                v
      results/report.md  +  results/*.png
      (Markdown tables + embedded chart images,
       deterministic given the same input CSV -
       fixed RNG seed for the bootstrap CI)
```

## Data flow 4: correctness check (`eval/human_agreement.py` + `eval/correctness_report.py`)

A separate branch from data flows 2-3 - it also starts from `eval/results.json` (step
2's output) but never touches `repeatability_raw.csv` or `report.py`. One HTTP round
trip per **labeled** ticket (all 25, not just the 5 frozen ones), to the **Jev**
model - each ticket scored once, not repeated.

```
eval/results.json                    eval/dataset.json
(frozen agent answers)                (human_label per ticket:
        |                              does_pass, quality, notes)
        +------------------+------------------+
                            v
                load_labeled_cases()
                -- all 25 tickets with a human_label
                            |
                            v
              +---------------------------+
              | eval/human_agreement.py     |
              | ThreadPoolExecutor           |
              | (--concurrency, default 10)  |
              |                               |
              | for each labeled ticket not   |
              | already cached:               |
              |   jev_judge.evaluate(case)    |  -- same evaluate() call as
              +-------------+-----------------+     repeatability.py uses
                            |
                            v
              results/human_agreement_calls.json
              (per-ticket cache: does_pass, quality,
               groundedness, search_behavior, usefulness)
                            |
                            v
              compute Cohen's kappa (does_pass) +
              Spearman/Pearson (quality) vs. each
              ticket's human_label, + confusion matrix
                            |
                            v
              results/human_agreement.json
              (aggregate stats + list of scored ticket_ids)
                            |
                            v
              +----------------------------+
              | eval/correctness_report.py   |
              |  -- no API calls, pure read   |
              |  joins human_agreement.json + |
              |  the per-ticket cache +       |
              |  dataset.json's human_label   |
              +-------------+----------------+
                            |
                            v
              results/correctness_report.md
              (agreement summary + confusion matrix +
               per-ticket human-vs-Jev table, sorted
               by disagreement - "is Jev right?")
```

## Cross-cutting: `eval/common.py`

Every arrow above that crosses into "OpenRouter API call" goes through the same two
helpers, and every judge/agent score goes through the same scoring math:

```
  retry_with_backoff(fn)              compute_quality(groundedness,
    - exponential backoff + jitter      search_behavior, usefulness)
    - retries on HTTP 429/500/502/      - search_behavior mapped via
      503/504 and connection errors       SEARCH_BEHAVIOR_SCORE:
    - wraps every agent call and          appropriate=1.0, unnecessary=0.5,
      every judge call                    failed=0.0
                                         - quality = mean of the three
  log_call(record)
    - appends one JSON line to        derive_does_pass(quality)
      results/calls.jsonl for every     - does_pass = 1 if quality >= 0.5
      success/retry/error, across         (DOES_PASS_THRESHOLD) else 0
      run_agent / repeatability /
      human_agreement / jev_judge
      (gitignored - local audit
      trail only)
```
