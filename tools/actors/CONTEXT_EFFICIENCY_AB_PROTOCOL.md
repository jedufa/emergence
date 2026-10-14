# EIEL Context-Efficiency A/B Validation Protocol

## Status

Manual protocol for the next slice. It is not an automated quota controller and
it grants no engineering authority.

## Question

Does the V1.9 Context Epoch workflow reduce Codex consumption per accepted Work
Package without increasing defects, failed reviews, or recovery failures?

## Baseline

Use the retained V1.9 forensic baseline generated from the supplied Codex JSONL
sessions:

- `verification/v1_9_context_efficiency_baseline.json`
- `verification/v1_9_context_efficiency_baseline.md`

## Treatment

For each treatment package:

1. create one small Work Package with read/output/budget contracts;
2. use one conversation for one authority phase;
3. start from `python -m tools.eiel start`;
4. read only the initial working set first;
5. redirect broad output and inspect summaries;
6. checkpoint and generate a Context Receipt at the epoch boundary;
7. continue review or qualification in a fresh conversation;
8. retain review outcome and accepted Work status.

## Minimum sample

Use at least six comparable packages when quota permits:

- two small mechanical implementation packages;
- two medium multi-file implementation packages;
- one repair cycle;
- one review or qualification phase.

Do not manufacture work merely to improve the metric.

## Measurements

Capture per package:

- duration;
- task starts;
- tool calls;
- context compactions;
- cumulative and cached input tokens;
- output tokens;
- raw shell-output tokens;
- broad-output events;
- authority-file reads;
- searches before first edit;
- review verdict;
- rework count;
- accepted or not accepted.

Primary measure:

> Accepted Work Packages per 10% of Pro allowance.

Supporting measures:

- cumulative input per accepted package;
- shell-output tokens per accepted package;
- compactions per accepted package;
- tool calls before first useful edit;
- first-pass review acceptance;
- fresh-session recovery success.

## Decision

- **A:** substantial efficiency improvement with no material quality regression;
- **B:** some improvement, continue manually and tune package defaults;
- **C:** no interpretable improvement because measurements are incomplete;
- **D:** workflow increases failures, hides evidence, or creates another authority.

Do not add automatic routing, hard limits, or a Context Manager before this
protocol produces real treatment evidence.

## V1.11 projection measurements

For each treatment package, also retain:

- full Workspace opening bytes and lines;
- compact Workspace opening bytes and lines;
- compact-to-full reduction percentage;
- whether compact output remained within `maximum_display_lines`;
- whether `--output` retained the complete payload;
- number and total bytes of declared initial files;
- largest declared initial files;
- whether global authority files were reread despite an unchanged supplied digest;
- whether the operator required `--full` before the first useful edit.

The V1.11 offline projection gate establishes presentation correctness and size
reduction only. It does not prove lower Pro allowance consumption. The original
six-package live treatment remains the decision gate for workflow-level claims.

## V1.16 additional observations

For each treatment package also record:

- whether routine actor commands used compact display plus saved full JSON;
- number of full expensive-model architecture passes in the programme;
- whether routine continuation crossed the Bridge;
- whether bounded U1/U2 repairs returned to Spark;
- whether global authority was reread without digest drift or a disputed section.

These observations remain manual and evidence-only. They do not authorize automatic routing or quotas.
