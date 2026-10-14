# V1.10.3 Real Codex Host Smoke Procedure

This procedure is the remaining host-only gate. Do not use valuable Work.

## Preconditions

- V1.10.3 integrated in a clean disposable worktree.
- current Codex CLI installed and authenticated;
- one tiny package owning one disposable fixture file;
- explicit human Dispatch Request;
- Scheduler limited to one Assignment.

## Procedure

1. Run `python -m tools.eiel assignment codex-capability`.
2. Confirm the probe reports `CODEX_EXEC_READY`.
3. Authorize one implementation Dispatch using `workspace-write` only.
4. Run `python -m tools.eiel scheduler run-once`.
5. Verify the Session was mechanically ACTIVE before the first repository edit.
6. Verify the actual `thread.started` ID replaced the provisional identity in
   both Lease and Session.
7. Verify the worker changed only owned files, committed, refreshed Bootstrap,
   attached it to the Session, and wrote one structured Worker Result.
8. Verify Assignment and Lease are CLOSED and runtime has forgotten the
   terminal process row.
9. Attempt an independent review with the same real thread ID and verify it is
   rejected.
10. Repeat with a disposable sleeping worker, terminate the Scheduler, and
    verify no process-group descendant survives.
11. Repeat with worker interruption, delete/disable Scheduler code, and resume
    the prepared/interrupted Session manually.

## Pass decision

`HOST-CODEX-SMOKE-PASS` requires every step above. Any failure keeps real
unattended execution disabled while manual operation and Fake Worker remain
available.

## Restart and containment extension

After the first disposable success, repeat with a worker that remains active
long enough to kill only the foreground Scheduler process. Restart
`scheduler start` against the same state root and prove:

- startup reconciliation adopts the ACK-gated wrapper;
- the Lease remains heartbeated;
- no second Assignment launches;
- the original structured result is submitted;
- wrapper and worker process groups are empty afterward;
- stdout and stderr remain within configured bounds.

This extension is mandatory for real unattended qualification.

## Result settlement and identity extension

The host gate must also establish the seams that cannot be proven by a fake
Codex executable:

1. Confirm `thread_identity.json` is written from the first valid
   `thread.started` event before the bounded human event log rotates.
2. Force more than the configured event-log limit and verify the same immutable
   thread identity remains available after rotation.
3. Complete a worker before Lease expiry, stop the Scheduler after the result is
   durable, wait until the Lease expires, then restart. The result must be
   accepted from its pre-expiry execution timestamps and the expired Lease must
   close without being revived.
4. In a disposable fixture only, expose a partial `--output-last-message` file,
   then complete it. Reconciliation must report `RESULT_PENDING` first and
   submit the completed result later.
5. Leave a stable malformed result. It must remain preserved with validation
   errors and end as explicit `RESULT_INVALID` plus worker loss.
6. Verify the real output file's actual write behavior. Record whether the
   installed Codex CLI uses atomic replacement; do not assume it from fake
   worker behavior.

These checks are mandatory for `HOST-CODEX-SMOKE-PASS`.
