# EIEL WRA-1 Estimator Rules

## Status

Human-maintained advisory rules. These rules are **not executable code** and are
not applied automatically by WRA-1.

## Governance

- Update only after reviewing immutable Work `token_usage.json` artifacts.
- Require an Estimator Review Report and explicit human approval.
- Preserve prior revisions in Git history.
- Never infer cost, credits, quota percentage, or plan consumption.
- Never change Work acceptance, authority, model selection, or Scheduler
  eligibility based on these estimates.
- Treat small samples as observations, not universal laws.

## Initial rough rules

These ranges are starting hypotheses from the WRA-1 implementation proposal,
not validated predictions.

| Work type | Initial expected token range | Confidence |
|---|---:|---|
| Implementation | 2,000,000–4,000,000 | none |
| Review | 300,000–900,000 | none |
| Qualification | 1,000,000–2,000,000 | none |

## Required evidence before refinement

For each proposed change, report at least:

- sample size;
- date window;
- Work type and subsystem coverage;
- models and reasoning levels represented;
- fresh versus resumed Context Epochs;
- measured, partial, and unavailable counts;
- median and spread of actual tokens;
- within-range fraction;
- recurring overestimation or underestimation;
- whether the proposed rule generalizes beyond one outlier.

## Change log

No evidence-derived rule changes have been approved yet.
