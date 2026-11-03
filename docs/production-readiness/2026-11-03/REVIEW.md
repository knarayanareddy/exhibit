# exhibit — review

## 1. Scope and verdict
Repository: `knarayanareddy/exhibit` at pinned SHA `1a7f79e236f4365e9203b1d958d3b7971da5029d`.

Stage: implementation prototype for an evidence-pack compiler/API, not a proven production service. Runtime NOT RUN.

Personally inspected evidence was limited to: `README.md:1-47`, `pyproject.toml:1-26`, `exhibit/main.py:1-220`, `exhibit/__main__.py:1-3`, plus two read-only own-SHA source fetches for `exhibit/api.py:1-110,145-175` and `exhibit/storage.py:1-110`. `exhibit/evaluator.py`, `exhibit/models.py`, `exhibit/pack_compiler.py`, `exhibit/policy.py`, `exhibit/standards.py`, and fixture contents under `fixtures/traces/` were not inspected. No target execution, installation, CLI run, API serving, fixture replay, signature verification, CI replay, browser/UI check, deployment, restore drill, or provider access was performed.

Verdict: the draft remains source-grounded on a bounded sample, but the repository still carries three substantive concerns: README-level overclaim risk around immutability/immunity, a probable dashboard/template-path gap, and missing observed automation/release evidence.

## 2. Architecture and product boundary
The sampled entry points show a CLI wrapper and a FastAPI-serving path, not a complete audited platform boundary. `exhibit/__main__.py:1-3` exits through `main()`, and `exhibit/main.py:14-17` imports `EvidenceRepository`, `Settings`, `create_app`, `heuristic_evaluate_trace`, and `ingest_jsonl`. The CLI exposes `serve` and `demo` subcommands, including local database/artifact paths and fixture ingestion defaults (`exhibit/main.py:40-96`). README architecture text describes a pipeline from `spans.jsonl` ingestion through evaluation, human oversight, and dossier compilation (`README.md:24-32`).

Within the sampled boundary, the visible product surface is:

- CLI startup and demo wiring in `exhibit/main.py`.
- FastAPI configuration and repository settings in `exhibit/api.py`.
- Receipt-chain and integrity primitives in `exhibit/storage.py`.

Outside the sampled boundary are the actual compiler logic, policy enforcement details, model schemas, evaluator behavior, and fixture semantics. The repository also advertises a possible dashboard/console, but UI behavior and accessibility were not exercised.

## 3. Findings ledger
### F01 — unverified risk, high — README claim scope exceeds demonstrated sample
- Evidence 1: [`README.md:3`](https://github.com/knarayanareddy/exhibit/blob/1a7f79e236f4365e9203b1d958d3b7971da5029d/README.md#L3-L3) — “`Exhibit produces immutable engineering artifacts.`”
- Evidence 2: [`README.md:11-19`](https://github.com/knarayanareddy/exhibit/blob/1a7f79e236f4365e9203b1d958d3b7971da5029d/README.md#L11-L19) — “`compiles an immutable exhibit.json dossier`” and “`Prompt injection immunity benchmarks (ex-inject-01)`”.

Impact: the reviewed sample shows integrity-oriented code and bounded limitations, but this static pass did not verify signer identity, external anchoring, tamper detection, or benchmark methodology. “Immutable” and “immunity” read stronger than the sampled evidence proves.

Recommended correction: narrow public wording to the exact demonstrated mechanism and record its limitations until an independent test set shows tamper detection, signer binding, and benchmark scope.

### F02 — confirmed static, medium — API defaults to a template path not present in the sampled tree inventory
- Evidence 1: [`exhibit/api.py:149-155`](https://github.com/knarayanareddy/exhibit/blob/1a7f79e236f4365e9203b1d958d3b7971da5029d/exhibit/api.py#L149-L155) — “`template_path: Path = Path("web/templates")`”.
- Evidence 2: local snapshot inventory `evidence/exhibit/snapshot.json:16-37` lists the full sampled tree and does not include any `web/` directory.

Impact: a dashboard/operator-console path is advertised, but the sampled repository tree does not show the matching template directory. An uninspected fallback may exist elsewhere, so this is a bounded static gap rather than proof of runtime failure.

Recommended correction: exercise the served UI in isolation, then either add the missing template assets or explicitly document that the supported operator surface is CLI/API only.

### F03 — confirmed static, medium — sampled evidence shows packaging for pytest but no sampled tests or CI workflows
- Evidence 1: [`pyproject.toml:22-25`](https://github.com/knarayanareddy/exhibit/blob/1a7f79e236f4365e9203b1d958d3b7971da5029d/pyproject.toml#L22-L25) — “`dev = [`” with “`pytest>=8.0.0`” and “`pytest-asyncio>=0.23.0`”.
- Evidence 2: [`README.md:42-43`](https://github.com/knarayanareddy/exhibit/blob/1a7f79e236f4365e9203b1d958d3b7971da5029d/README.md#L42-L43) — “`# Run test suite`” and “`pytest`”.
- Evidence 3: local snapshot inventory `evidence/exhibit/snapshot.json:16-37` shows no sampled `tests/` tree and no `.github/workflows/` entries; `recent_runs` is empty at `snapshot.json:69-70`.

Impact: the prototype advertises tests, but this evidence bundle does not show actual test sources or CI history. That leaves immutability, upload-boundary, auth, and restore claims without observed automation support.

Recommended correction: add repository-visible tests and CI, and require negative tests that fail on tampered evidence, missing auth, or unsupported UI assumptions.

## 4. Five-discipline assessment
- **Correctness / code review:** `exhibit/storage.py:41-87` defines receipt-chain contexts and `InvalidHumanReviewError`, which is consistent with integrity intent, but the actual compiler and model binding paths were not inspected.
- **Automation / testing:** the manifest and README both reference pytest, yet the sampled inventory contains no tests or Actions workflows. Negative-path automation is therefore unverified.
- **DevOps / operations:** `exhibit/main.py:55-64` and `exhibit/api.py:151-160` show local SQLite/artifact/key settings, which implies backup, key escrow, and restore obligations. No observed restore or release procedure was present in the sample.
- **OWASP / security:** `exhibit/api.py:174-185` bounds upload/key configuration, and `exhibit/storage.py:82-87` rejects invalid human review records, but this pass did not verify route authentication, authorization, secret handling, or provider isolation.
- **UI / UX / accessibility:** `exhibit/main.py:40-43` advertises a “FastAPI compliance dashboard,” but the sampled tree lacks `web/templates/`, and no UI or accessibility behavior was exercised.

## 5. Production blockers and owner questions
Blockers before any stronger release or public claim:

1. Who owns legal/counsel approval for statutory wording and the repository’s non-certification boundary?
2. What signer identity model, key custody, and rotation policy make “human reviewer sign-off” meaningful in production?
3. Is a browser dashboard actually a supported surface, or is the supported boundary CLI/API only?
4. What hosting, backup, restore, and rollback path protects SQLite records, artifact files, and verification keys together?
5. Will LLM/provider-backed evaluation ever be enabled, and if so under what approval and data-handling controls?

No certification, conformance determination, restore readiness, deployment readiness, or accessibility readiness is established by this review.

## 6. Evidence inventory and limitations
Inspected paths/ranges:

- `fleet-hardening-review/AUTHOR_CONTRACT.md:1-30`
- `fleet-hardening-review/evidence/exhibit/snapshot.json:1-71`
- `fleet-hardening-review/evidence/exhibit/numbered-sources/README.md.txt:1-47`
- `fleet-hardening-review/evidence/exhibit/numbered-sources/pyproject.toml.txt:1-26`
- `fleet-hardening-review/evidence/exhibit/numbered-sources/exhibit/main.py.txt:1-220`
- `fleet-hardening-review/evidence/exhibit/numbered-sources/exhibit/__main__.py.txt:1-3`
- own-SHA read-only source fetch: `exhibit/api.py:1-110,145-175`
- own-SHA read-only source fetch: `exhibit/storage.py:1-110`

Explicitly unrun checks:

- No install or import of the target package.
- No `pytest`, no CI replay, no GitHub Actions validation.
- No FastAPI serve, browser, template rendering, or accessibility inspection.
- No signature verification, tamper test, restore drill, backup exercise, or rollback exercise.
- No live provider/API key use, no remote writes, and no publication.

These conclusions apply only to the sampled ranges above and should not be read as whole-repository absence claims outside those boundaries.
