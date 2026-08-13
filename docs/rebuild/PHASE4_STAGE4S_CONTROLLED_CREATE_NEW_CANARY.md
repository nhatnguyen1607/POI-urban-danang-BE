# Phase 4 Stage 4S - Controlled CREATE_NEW Canary

## Status

`PASS_APPLIED_TO_FEATURE_BRANCH`

Stage 4S applied the first bounded, human-approved `CREATE_NEW` canary on the
feature branch only. It did not enable automatic creation, merge or delete an
entity, mutate an existing canonical row, expose the enrichment sidecar at
runtime, or touch a production database or Firebase.

## Human Decisions

The authoritative Stage 4R decision CSV contains 26 fully resolved case IDs:
7 `APPROVE`, 4 `REJECT`, and 15 `DEFER`. Prefix, full case ID, expected name,
and decision counts were validated before planning. Decision source is `HUMAN`
and scope is `CREATE_NEW_CANARY`; approval is bound to the reviewed case and
fingerprint and is not reusable as general policy authorization.

All seven approvals survived fingerprint, current-canonical absence,
source-duplicate, pairwise-duplicate, traveler relevance, category, polygon,
coordinate, provenance, and license checks. No approved case was dropped, and
no rejected or deferred case entered the apply plan.

## Applied Cases

| POI | Canonical ID |
| --- | --- |
| Khach San Golden Sea 3 | `candidate:da-nang:44f30f9baa6b2340` |
| Chi House Danang | `candidate:da-nang:2a5e1c00c8f0bacf` |
| Kurumi | `candidate:da-nang:671afec933a1917a` |
| Di tich danh thang Ngu Hanh Son | `candidate:da-nang:c017ebe6c5f87ec2` |
| Ocean Villas | `candidate:da-nang:275459616eaeffcc` |
| Banh mi & Cafe 43 | `candidate:da-nang:b7de1964b4ee075e` |
| Alani Sea View Hotel | `candidate:da-nang:db67529b9f1809c9` |

IDs use the deterministic `candidate:da-nang:<source-identity-hash>` policy.
They do not depend on row order.

## Canonical And Sidecar Result

- Canonical rows: 4,166 -> 4,173.
- SHA-256 before:
  `39647b29308813a7ec19e4695fd2b95ffa27743db8e8cb46800c45d3a3fe6ded`.
- SHA-256 after:
  `dcb404cc8b5c7a9b5fd70df63039ab8f828c504270e22b12a671fa4ed61583f4`.
- Existing canonical bytes changed: 0; rows appended: 7; rows deleted: 0.
- Canonical IDs: 4,173 unique; unresolved new-to-existing or new-to-new
  duplicates: 0.
- Sidecar records created: 19; duplicate identities: 0; orphans: 0.
- Logical sidecar hash before:
  `2bb10fa42c4be338eea7a034cdab1060c1760fb1159887d8008af9c02560d7bc`.
- Logical sidecar hash after:
  `5c65e686665b22faa6a11f57053acc7121a784bf310fa7a812e938792906333c`.

Canonical rows contain only supported core fields. Additional source fields
remain in the non-runtime sidecar with field lineage, source identity, license,
attribution, processing version, and policy version. Google data is absent.

## Safety And Compatibility

The apply command defaults to dry-run. A real feature-branch write requires
both `--apply` and `URBANAGENT_ALLOW_STAGE4S_CANONICAL_WRITE=true`. The plan
contains only exact human approvals and records zero existing-row mutations,
deletes, merges, and policy-only creates.

Rollback restored the baseline canonical bytes and sidecar exactly. Replaying
the same baseline, decisions, evidence, and policy produced the same surviving
cases, IDs, order, plan hash, candidate canonical SHA, and sidecar hash.

The canonical loader reports 4,173 valid POIs with no duplicate ID or invalid
coordinate. Recommendation and trip-preview smokes are nonempty. Traveler API
code and its external-source request path were not changed.

## Scheduler Separation

The existing `UrbanAgent POI Scheduled Sync` task remains enabled and ready;
its last result is 0. Scheduler cadence was not changed. Stage 4S is separate
from the Stage 4M executor, and `AUTO_CREATE_NEW` remains false.

## Validation

Focused Stage 4H-4S tests: 213 total, 204 passed, 0 failed, and 9 guarded
Stage 4K tests skipped. Stage 4S-specific tests include decision validation,
all safety gates, exact append behavior, sidecar integrity, rollback,
determinism, runtime loading, recommendation, and trip preview.

## Proposed Stage 4T

Propose only `POST-MERGE CREATE_NEW CANARY VERIFICATION + PHASE 4 CLOSEOUT`.
First review and merge this bounded diff, then verify the canonical baseline,
runtime compatibility, scheduler separation, duplicate safety, and scoped
decision-memory behavior. Do not increase the batch size or enable automatic
`CREATE_NEW` during Stage 4T.
