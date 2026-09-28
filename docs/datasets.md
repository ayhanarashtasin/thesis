# Dataset and evaluation protocol

Phase 1 fixes dataset identity, role allocation, split construction, prompt generation, and
reward semantics before training is introduced. Configuration is authoritative; the checked-in
JSON manifests record the exact memberships produced from those settings without storing raw
dataset records.

## Pinned sources and permitted roles

| Project ID | Source | Revision | Project roles |
| --- | --- | --- | --- |
| `gsm8k` | `openai/gsm8k` / `main` | `740312add88f781978c0658806c59bc2815b9866` | train, validation, anchor candidate, held-out test |
| `arc_challenge` | `allenai/ai2_arc` / `ARC-Challenge` | `210d026faf9955653af8916fad021475a3f00453` | train, validation, anchor candidate, held-out test |
| `mbpp` | `google-research-datasets/mbpp` / `sanitized` | `4bb6404fdc6cacfda99d4ac4205087b89d32030c` | train, validation, anchor candidate, held-out test |
| `constraints` | generated `rahc_lora/verifiable_constraints` | `1.0.0` | train, validation, anchor candidate, held-out custom test |
| `ifeval` | `google/IFEval` | `966cd89545d6b6acfd7638bc708b98261ca58e84` | evaluation only |
| `mmlu` | `cais/mmlu` / `all` | `c30699e8356da336a370243923dbaf21066bb9fe` | evaluation only; fixed configured subjects |
| `hellaswag` | `Rowan/hellaswag` | `218ec52e09a7e7462a5400043bb9a69a41d06b76` | evaluation only |
| `arc_easy` | `allenai/ai2_arc` / `ARC-Easy` | `210d026faf9955653af8916fad021475a3f00453` | evaluation only |
| `perplexity` | `Salesforce/wikitext` / `wikitext-2-raw-v1` | `b08601e04326c79dfdd32d625aee71d232d685c3` | evaluation only; nonempty test records |

The fixed MMLU subset contains `abstract_algebra`, `college_computer_science`,
`high_school_physics`, and `professional_medicine`. Evaluation-only adapters reject training,
validation/tuning, and anchor-candidate access. Official IFEval is never exposed through a
training role and requires a version-matched official scorer; there is no approximate fallback.

## Deterministic split construction

Source identities are SHA-256 hashes over configured immutable identity fields. Train-source
records are ordered by SHA-256 of `split_seed:example_id`, which makes role assignment independent
of provider iteration order.

- GSM8K derives validation and anchor candidates from its source `train` split, then uses the
  official `test` split only as held-out test.
- ARC-Challenge and MBPP preserve their official validation and test splits. Anchor candidates are
  deterministically removed from source `train` before the remaining records become project train.
- Generated constraints use distinct template families for train, validation, anchor candidate,
  and test roles; prompts and IDs are unique across all roles.
- Evaluation datasets retain their configured held-out source split under the sole
  `evaluation_only` role.

Every trainable manifest enforces nonempty roles and rejects repeated example IDs or exact content
hashes across roles. See [ADR 0001](decisions/0001-task-split-construction.md) for the rationale and
scientific consequences.

## Current manifest memberships

| Dataset | Membership counts |
| --- | --- |
| GSM8K | train 5,979; validation 747; anchor candidate 747; test 1,319 |
| ARC-Challenge | train 1,007; validation 299; anchor candidate 112; test 1,172 |
| MBPP | train 108; validation 43; anchor candidate 12; test 257 |
| Custom constraints | train 32; validation 16; anchor candidate 16; test 16 |
| Official IFEval | evaluation only 541 |
| Fixed MMLU subset | evaluation only 623 |
| HellaSwag | evaluation only 10,042 |
| ARC-Easy | evaluation only 2,376 |
| WikiText-2 held-out corpus | evaluation only 2,891 nonempty records |

Each manifest includes the canonical source, immutable revision, source-split mapping, licenses and
citations, preprocessing/prompt/reward versions, exact role membership IDs, content hashes, and
per-example provenance. The generated-constraint manifest additionally records grammar, template,
validator, dataset, and generation-seed versions plus template family, constraint types, and prompt
hashes for each example.

## Custom constraints and IFEval isolation

The constraint generator is deterministic for its configured seed and currently emits 80 examples
from eight versioned template families. Preparation compares every custom prompt with official
IFEval using both exact text and normalized text. The checked-in audit reports zero exact and zero
normalized overlaps between 80 custom prompts and 541 official prompts. Any overlap raises an error
before manifests are written.

## Deterministic rewards and generated code

Reward adapters cover numeric/text exact match, unambiguous multiple choice, fraction of satisfied
instruction constraints, and fraction of hidden MBPP tests passed. Malformed or ambiguous answers
receive the documented zero score rather than a guessed interpretation.

MBPP responses are executed only through the configured Docker backend. It uses a digest-pinned
Python image, no network, a read-only root, dropped capabilities, `no-new-privileges`, an
unprivileged user, and CPU, memory, process, file-descriptor, time, and output limits. The input
workspace is mounted read-only and the container is removed on timeout or output overflow. If
Docker or its daemon is unavailable, scoring fails closed; there is no host-subprocess fallback.

## Rebuilding manifests

With the pinned datasets available from Hugging Face or already present in the configured cache:

```powershell
python -m rahc_lora.cli prepare-data
```

For a cache-only repeat, set `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` before running the same
command. Preparation writes only JSON manifests and overlap evidence under `data/manifests/`; raw
datasets remain in the external cache and must not be committed.
