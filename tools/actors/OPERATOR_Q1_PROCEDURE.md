# EIEL OPERATOR-Q1 Procedure

## Purpose

Qualify the repository-local operator kernel in bounded fresh processes. This
procedure covers Work Packages, Work, Session, Bootstrap, Workspace, authority
routing, review/repair/promotion, optional Bridge behavior, and the V1.11 operator hardening and compact-projection contract.

It does not qualify real Codex execution, scientific simulation mechanics, or
campaign evidence claims.

## Command

From the repository root:

```bash
python -m tools.verification.verify_operator_q1 \
  --output tools/actors/verification/v1_11_operator_q1_ledger.json
```

Each frozen target runs in a separate process with a bounded deadline. The
runner prints progress, terminates the complete test process group on timeout,
produces one aggregate ledger, and exits nonzero on any failure or timeout.

## Pass criteria

- every frozen target passes;
- no target times out;
- `git diff --check` passes separately;
- all actor JSON schemas validate as Draft 2020-12;
- the exact transfer ZIP is cleanly extracted and its manual CLI surfaces run
  without the Scheduler package.

## Failure handling

A failed or timed-out target blocks OPERATOR-Q1. Do not replace a failing target,
weaken its assertion, or declare the retained ledger sufficient. Repair the
bounded defect, restart the full frozen run, and preserve the failed attempt in
release notes when it affected the decision.
