# Jev-as-Judge: Correctness vs. Human Labels

Generated from 25 labeled ticket(s) in `eval/dataset.json`, each scored once by Jev and compared against a hand-written `human_label`. This answers a different question from `report.md`: not "is Jev consistent?" but "is Jev *right*?"

## Agreement summary

|   n_labeled |   cohen_kappa_does_pass |   spearman_rho_quality |   spearman_p_value |   pearson_r_quality |   pearson_p_value |
|------------:|------------------------:|-----------------------:|-------------------:|--------------------:|------------------:|
|          25 |                       0 |                 0.5471 |             0.0047 |              0.3282 |            0.1093 |

**Confusion matrix** (rows=human does_pass, cols=Jev does_pass, labels [0, 1]):

```
[0, 3]
[0, 22]
```

## Per-ticket comparison (3 does_pass mismatch(es), sorted by quality delta)

| ticket_id   |   human_does_pass |   jev_does_pass | match   |   human_quality |   jev_quality |   quality_delta |
|:------------|------------------:|----------------:|:--------|----------------:|--------------:|----------------:|
| t17         |                 0 |               1 | NO      |             0   |        0.9    |          0.9    |
| t19         |                 0 |               1 | NO      |             0   |        0.8233 |          0.8233 |
| t18         |                 0 |               1 | NO      |             0   |        0.6033 |          0.6033 |
| t22         |                 1 |               1 | yes     |             0.9 |        0.6133 |          0.2867 |
| t20         |                 1 |               1 | yes     |             1   |        0.7433 |          0.2567 |
| t25         |                 1 |               1 | yes     |             0.8 |        0.58   |          0.22   |
| t23         |                 1 |               1 | yes     |             0.7 |        0.8667 |          0.1667 |
| t01         |                 1 |               1 | yes     |             1   |        0.8433 |          0.1567 |
| t21         |                 1 |               1 | yes     |             0.8 |        0.6933 |          0.1067 |
| t07         |                 1 |               1 | yes     |             1   |        0.8967 |          0.1033 |
| t16         |                 1 |               1 | yes     |             1   |        0.9    |          0.1    |
| t06         |                 1 |               1 | yes     |             0.9 |        0.8033 |          0.0967 |
| t14         |                 1 |               1 | yes     |             1   |        0.93   |          0.07   |
| t10         |                 1 |               1 | yes     |             1   |        0.93   |          0.07   |
| t03         |                 1 |               1 | yes     |             1   |        0.9367 |          0.0633 |
| t02         |                 1 |               1 | yes     |             0.9 |        0.8367 |          0.0633 |
| t08         |                 1 |               1 | yes     |             1   |        0.94   |          0.06   |
| t13         |                 1 |               1 | yes     |             0.9 |        0.96   |          0.06   |
| t11         |                 1 |               1 | yes     |             0.9 |        0.9433 |          0.0433 |
| t24         |                 1 |               1 | yes     |             0.8 |        0.7567 |          0.0433 |
| t12         |                 1 |               1 | yes     |             1   |        0.96   |          0.04   |
| t05         |                 1 |               1 | yes     |             1   |        0.9667 |          0.0333 |
| t09         |                 1 |               1 | yes     |             0.9 |        0.8767 |          0.0233 |
| t04         |                 1 |               1 | yes     |             0.9 |        0.8833 |          0.0167 |
| t15         |                 1 |               1 | yes     |             0.9 |        0.8867 |          0.0133 |

## What this means

- **Cohen's kappa** measures does_pass agreement beyond chance: <0 no agreement, 0-0.2 slight, 0.2-0.4 fair, 0.4-0.6 moderate, 0.6-0.8 substantial, 0.8-1.0 near-perfect. It's null above when either the human labels or Jev's verdicts have no variance to measure agreement on.

- **Spearman/Pearson on quality** measure how well Jev's continuous quality score tracks the human's, independent of the does_pass threshold - a high correlation with a low kappa would mean Jev ranks tickets similarly to a human but draws the pass/fail line in a different place.

- **Small sample size.** 25 labeled tickets is enough to spot gross disagreement, not to certify Jev statistically - treat this as a smoke test for correctness, not a benchmark.

- **One call per ticket, not repeated.** Unlike `report.md`'s repeatability sweep, each ticket here is scored once, so an unlucky single sample isn't averaged out. `human_agreement.py` caches per-ticket results in `human_agreement_calls.json` and won't re-score a cached ticket - delete its entry from that file to force a re-score.
