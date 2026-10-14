# Work Package Review and Repair Example

**Purpose:** Show the V1.9 package-bound implementation, review disagreement,
bounded repair, rereview, and Work acceptance path.

## 1. Implementation delivery

The Implementation Authority Session works from the exact bound Work Package,
commits only owned files, leaves a clean worktree, and completes:

```text
python -m tools.eiel sessionManager complete SES-IMPLEMENTATION-1
```

The Session records `IMPLEMENTATION_COMPLETE`, the exact implementation commit,
changed files, package version, and canonical package digest. Work remains open.

## 2. Substantive review

A fresh Review Authority Session explicitly targets the delivery Session:

```text
python -m tools.eiel sessionManager start WORK-042 \
  --bootstrap-record BST-CURRENT \
  --codex-id review-context-1 \
  --repo-root /tmp/emergence-example/user/emergence-worktrees/work-042 \
  --role REVIEW_AUTHORITY \
  --review-target-session SES-IMPLEMENTATION-1 \
  --model "GPT-5.5 High" \
  --reasoning-level HIGH
```

The same model family may be used. Prefer a fresh, separate Codex identity when
immediately available. If it is not, the same-identity reduced-assurance review
must still evaluate the same evidence and record
`REVIEWER_INDEPENDENCE_UNAVAILABLE`; reviewer identity never blocks acceptance.

## 3. Changes required inside the same package

```text
python -m tools.eiel sessionManager complete SES-REVIEW-1 \
  --review-verdict CHANGES_REQUIRED \
  --review-finding "The error branch does not fail closed." \
  --repair-disposition WITHIN_PACKAGE
```

This verdict does not accept Work. A new Implementation Authority Session performs
the bounded repair under the unchanged package. The original Session is not
reopened.

## 4. Repair delivery

```text
python -m tools.eiel sessionManager start WORK-042 \
  --bootstrap-record BST-REPAIR \
  --codex-id spark-repair-context \
  --repo-root /tmp/emergence-example/user/emergence-worktrees/work-042 \
  --role IMPLEMENTATION_AUTHORITY \
  --model "GPT-5.3 Spark High" \
  --reasoning-level HIGH

python -m tools.eiel sessionManager complete SES-REPAIR-1
```

The repair commit becomes the latest delivery. Every review of an older commit is
now stale and cannot authorize completion.

## 5. Rereview and acceptance

```text
python -m tools.eiel sessionManager start WORK-042 \
  --bootstrap-record BST-REVIEW-2 \
  --codex-id review-context-2 \
  --repo-root /tmp/emergence-example/user/emergence-worktrees/work-042 \
  --role REVIEW_AUTHORITY \
  --review-target-session SES-REPAIR-1 \
  --model "GPT-5.5 High" \
  --reasoning-level HIGH

python -m tools.eiel sessionManager complete SES-REVIEW-2 \
  --review-verdict ACCEPT \
  --tests-reviewed "pytest -q tests/test_feature.py" \
  --evidence-reviewed "clean repair commit and focused test output" \
  --known-limitation "bounded package scope only"

python -m tools.eiel workManager complete WORK-042 --review-session SES-REVIEW-2
```

Only the accepted review of the latest delivery permits Work to reach
`COMPLETED`.

## When the package must change

Use `PROMOTE` and `REVISED_PACKAGE_REQUIRED`, not bounded repair, when review or
implementation discovers a change to:

- goal or scope;
- owned or forbidden files;
- dependencies;
- acceptance criteria;
- required tests or evidence;
- stop conditions;
- authority;
- uncertainty class.

The revised package uses a higher `package_version` and a new repository path.
Historical Sessions remain bound to the old digest.
