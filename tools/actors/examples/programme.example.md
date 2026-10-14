# Example Development Programme

**Status:** Planning guidance, not an executable schema  
**Root objective:** Add a bounded operator capability without creating a new management layer.

## Authority path

Architecture Authority -> Engineering Authority -> Implementation Authority -> Review Authority

## Work order

| Work | Purpose | Dependencies | May run in parallel with |
|---|---|---|---|
| WORK-010 | Shared contract | None | None |
| WORK-020 | Component A | WORK-010 | WORK-021 |
| WORK-021 | Component B | WORK-010 | WORK-020 |
| WORK-030 | Consolidate A and B | WORK-020, WORK-021 | None |
| WORK-040 | Independent review and closeout | WORK-030 | None |

## Manual implementation lanes

- Spark terminal 1: WORK-020
- Spark terminal 2: WORK-021
- Engineering Authority: WORK-030
- Review Authority: WORK-040

## Operating rules

- The human assigns terminals and models.
- Each implementation Session executes one bound Work Package.
- Implementation stops and requests promotion when uncertainty rises.
- Consolidation and integration are ordinary Work Items with their own packages.
- Implementation completion does not accept Work.
- A separate Review Authority Session records the verdict.
