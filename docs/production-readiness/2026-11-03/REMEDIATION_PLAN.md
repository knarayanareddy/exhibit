# exhibit — remediation plan

## 1. Phased roadmap with entry and exit criteria
| Phase | Focus | Entry criteria | Exit criteria |
| --- | --- | --- | --- |
| P0 | Baseline claim boundary | Current prototype; README and manifest make stronger claims than this sample proves | Claim matrix ties each public statement to inspected code/tests or is narrowed with explicit limitations |
| P1 | Supported entry points and automation | P0 complete; owner agrees intended product surface | CLI/API/UI behavior is observed in isolation and negative-path tests/CI exist |
| P2 | Deployment, recovery, and release governance | P1 complete; hosting and key-ownership decisions exist | Backup/restore, rollback, and release gates are written and observed on synthetic evidence |
| P3 | Independent acceptance | P2 artifacts exist | An independent reviewer observes failed negative tests and restore evidence before any stronger release claim |

## 2. Tasks
### T01 — Trace and narrow evidence claims (`F01`, `F03`)
**Prerequisites:** owner, legal/counsel delegate, security reviewer.

**Path labels:**
- **Inspected now:** `README.md`, `pyproject.toml`, `exhibit/main.py`, targeted `exhibit/api.py`, targeted `exhibit/storage.py`.
- **Tree-only existing paths to inspect during this task:** `exhibit/models.py`, `exhibit/pack_compiler.py`, `fixtures/traces/*.jsonl`.
- **Proposed future paths:** `tests/`, `.github/workflows/`, claim/limitations documentation.

1. Inspect the currently tree-only compiler/model/fixture paths and map each public claim about “immutable” artifacts and “immunity” benchmarks to a concrete mechanism, test, or limitation.
2. Define the trusted signer model, key rotation rules, and tamper-detection expectations that must hold before those claims can remain in release text.
3. Rewrite README and operator-facing wording so every surviving claim is narrower than or equal to the demonstrated evidence boundary.

**Expected deliverable:** claim matrix plus approved wording/limitations update.

**Verification procedure:** deliberately alter signed bytes, signer identity, or benchmark assumptions in planned negative tests and confirm the tests fail; obtain counsel review for the resulting wording.

**Definition of done:**
- [ ] Every claim has a backing mechanism or an explicit limitation.
- [ ] Wrong-key and tampered-evidence cases are specified to fail closed.
- [ ] Legal/non-certification wording is approved by the named owner delegate.

### T02 — Confirm supported entry points and add meaningful automation (`F02`, `F03`)
**Prerequisites:** T01.

**Path labels:**
- **Inspected now:** `exhibit/__main__.py`, `exhibit/main.py`, targeted `exhibit/api.py`, `pyproject.toml`.
- **Tree-only existing paths to inspect during this task:** `exhibit/evaluator.py`, `fixtures/traces/*.jsonl`.
- **Proposed future paths:** `web/templates/` if UI is supported, `tests/`, `.github/workflows/`.

1. Run the CLI/API in an isolated environment and determine whether the dashboard truly requires `web/templates/` or whether the supported operator surface is intentionally CLI/API only.
2. Add negative-path tests for upload bounds, malformed records, auth/authz expectations, and any template/UI startup assumption that must hold.
3. Add CI that executes those behavioral tests and records failure on deliberately broken behavior rather than only proving importability.

**Expected deliverable:** documented supported entry points plus repository-visible tests and CI.

**Verification procedure:** remove the template path, corrupt an upload, or disable an auth guard in a controlled branch and confirm at least one independent test fails.

**Definition of done:**
- [ ] Supported UI/API/CLI scope is recorded explicitly.
- [ ] Negative tests fail when core assumptions are broken.
- [ ] CI evidence exists for the agreed automation gate.

### T03 — Establish owner approvals, release gates, rollback, and recovery acceptance (preventive)
**Prerequisites:** T01 and T02, plus hosting approval.

**Path labels:**
- **Inspected now:** `exhibit/main.py`, targeted `exhibit/api.py`, targeted `exhibit/storage.py`.
- **Tree-only existing paths to inspect during this task:** `fixtures/standards/harmonized_standards.json`, `fixtures/traces/*.jsonl`.
- **Proposed future paths:** operations runbook, backup/restore procedures, release checklist, deployment manifests.

1. Assign owners for hosting, signing-key custody, retention/deletion, provider usage, and release authority, and record which changes require explicit approval.
2. Write a backup/restore and rollback procedure covering SQLite state, artifact files, and verification keys together, with a compatibility check for schema and signing material.
3. Perform and record an actual restore of synthetic evidence plus a rollback rehearsal before any public release claims stronger than the current prototype boundary.

**Expected deliverable:** owner-approved release/recovery runbook with observed synthetic restore evidence.

**Verification procedure:** restore a synthetic pack into an isolated environment, compare hashes/signatures to pre-backup values, then rehearse rollback to the previous release artifact and record the result.

**Definition of done:**
- [ ] Named owners approve hosting, key custody, retention, and provider posture.
- [ ] Restore evidence is observed and recorded; not presumed.
- [ ] Rollback instructions are tested against a prior release candidate.

## 3. Task dependency map
| Task | Depends on | Unlocks |
| --- | --- | --- |
| T01 | owner + counsel/security availability | claim-safe documentation and test targets |
| T02 | T01 | supported entry-point definition and automation gate |
| T03 | T01, T02, hosting approval | release/recovery acceptance |

Flow: `T01 -> T02 -> T03`, with no public claim expansion or release sign-off before T03 acceptance.

## 4. Production acceptance checklist
- [ ] Owner approves the non-certification boundary and narrowed public wording.
- [ ] Supported product surface is explicit: dashboard supported with assets, or CLI/API only.
- [ ] Negative tests are observed failing on tampered evidence or broken guards.
- [ ] CI is observed enforcing those tests.
- [ ] Backup, restore, and rollback are observed on synthetic evidence.
- [ ] Signing-key custody, rotation, and escrow are approved.
- [ ] Any provider-backed evaluation remains disabled or is explicitly approved with data-handling controls.
- [ ] Release notes preserve current limitations and do not claim legal conformity or certification.

## 5. Changes requiring owner approval
- Any wording that implies legal classification, conformity determination, or “immunity”.
- Any production hosting or public dashboard exposure.
- Any non-local provider/LLM evaluation path.
- Any signing-key custody, escrow, or rotation design.
- Any retention/deletion policy for traces, receipts, or artifacts.

## 6. Release, rollback, and recovery policy
- **Release gate:** no public release that strengthens current claims until T01-T03 are complete and independently observed.
- **Rollback policy:** retain the previous release artifact, matching verification keys, and schema-compatibility notes so rollback does not orphan stored evidence.
- **Recovery policy:** backup and restore must cover SQLite records, exported artifacts, and verification/signing material together; a restore is not accepted until hash and signature checks match pre-backup values.
- **Failure rule:** any failed negative test, missing restore evidence, or unresolved owner approval blocks release.

## 7. Independent verification requirements
Planned verification must be genuinely adversarial:

1. Break behavior on purpose: tamper with evidence, remove a required template or auth guard, or swap keys.
2. Confirm the relevant automated test or release gate fails for the broken case.
3. Observe backup restore and rollback on synthetic evidence before declaring recovery readiness.

Nothing in this plan should be read as already executed; all commands, tests, deployments, restores, and releases remain future work.
