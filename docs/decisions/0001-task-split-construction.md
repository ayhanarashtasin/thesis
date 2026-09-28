# ADR 0001: Deterministic task split construction

- Status: Accepted
- Date: 2026-09-22
- Author: RAHC-LoRA project

## Context

The continual experiment needs disjoint training, validation, anchor-candidate, and held-out test
roles while preserving official dataset splits where they exist. GSM8K has no official validation
split. ARC-Challenge and sanitized MBPP do. Anchor candidates must originate only from training
data, and exact membership must remain reproducible if a dataset provider changes iteration order.
Official IFEval and the fixed general-capability suite must remain evaluation-only.

## Decision

Use immutable dataset revisions and hash-derived source IDs. Deterministically rank every source
training record by SHA-256 of `split_seed:example_id`.

- When an official validation split exists, retain it as project validation. Remove the configured
  fraction of ranked source-training records for anchor candidates and use the remainder for train.
- When no official validation split exists, remove the configured validation fraction first, then
  the configured anchor fraction, from the same ranked source-training records. Use the remainder
  for train.
- Preserve the official source test split as held-out project test.
- Generate the custom-constraint roles from separate template-family allocations, with explicit
  grammar, template, validator, dataset, and seed versions.
- Give held-out capability corpora only an `evaluation_only` membership and expose no train or
  anchor API.
- Store stable IDs, exact content fingerprints, role membership, and provenance in manifests, but
  never store raw records in Git.

The concrete split settings live in Hydra configuration and are validated before data access.

## Alternatives considered

1. Use provider order with a seeded pseudo-random shuffle. Rejected because provider ordering can
   change and make the same seed produce different memberships.
2. Merge official validation into training and re-split everything. Rejected because it discards a
   source-defined evaluation boundary without scientific benefit.
3. Draw anchors from validation or test. Rejected because it leaks evaluation data into retention
   state and invalidates the continual-learning protocol.
4. Use official IFEval as custom-constraint training data or template inspiration. Rejected because
   IFEval is evaluation-only and would create direct benchmark contamination.

## Scientific consequences

- Role membership is deterministic, auditable, and independent of record enumeration order.
- Validation, anchor selection, and final testing remain isolated.
- Anchor candidates are eligible training-origin records, not held-out evaluation examples.
- Official splits are preserved wherever available, so source benchmark semantics remain visible.
- Exact and normalized custom/IFEval overlap checks provide a fail-fast contamination control.
- Changing identity fields, fractions, seeds, revisions, or preprocessing versions defines a new
  data protocol and requires regenerated manifests; results from different protocols must not be
  pooled silently.

## Engineering consequences

- Preparation must load pinned source splits, validate all prospective manifests, and write only
  atomic JSON artifacts.
- Trainable manifests reject duplicate IDs and exact content hashes across roles.
- Evaluation-only duplicate source identities receive deterministic occurrence suffixes while
  retaining occurrence provenance.
- Unit tests use an in-memory provider, so no import-time or test-time network access is required.

## Validation

- Unit tests cover stable splitting, disjoint memberships, manifest round-trips, invalid topology,
  duplicate content rejection, and evaluation-only access controls.
- Integration tests prepare manifests through the mockable provider and verify no raw dataset files
  are written.
- Source-backed manifests record all exact memberships and the custom/IFEval overlap audit.
