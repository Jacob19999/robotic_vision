<!--
Sync Impact Report
- Version change: template -> 1.0.0
- Modified principles:
  - Template Principle 1 -> I. Phase 1 Scope Discipline
  - Template Principle 2 -> II. Hardware-Aware Experimentation
  - Template Principle 3 -> III. Dataset Integrity and Evaluation Isolation
  - Template Principle 4 -> IV. Reproducible, Measurable Baselines
  - Template Principle 5 -> V. Incremental Decision Gates
- Added sections:
  - Current Phase Constraints
  - Workflow and Quality Gates
- Removed sections: none
- Templates requiring updates:
  - .specify/templates/plan-template.md -> updated
  - .specify/templates/spec-template.md -> updated
  - .specify/templates/tasks-template.md -> updated
  - .specify/templates/commands/ -> pending (directory not present in this project)
  - README.md -> already aligned; no change required
- Follow-up TODOs: none
-->

# Robotic Vision Constitution

## Core Principles

### I. Phase 1 Scope Discipline
All implementation work MUST support Phase 1 baseline development on real-world image
data only. NVIDIA Omniverse, Isaac Sim, synthetic data generation, mixed real plus
synthetic retraining, and deployment-stage optimizations are out of scope unless this
constitution is amended. Specs, plans, and tasks MAY mention later phases only as deferred
follow-up work and MUST NOT treat them as current implementation requirements.

Rationale: the project succeeds only if the real-data baseline is established first and the
team avoids solving later-phase problems prematurely.

### II. Hardware-Aware Experimentation
Every model, dataset workflow, and training configuration MUST be feasible on the local
development workstation: RTX 5070 with 12 GB VRAM, Ryzen 9 9900X, and 32 GB RAM. Full
fine-tuning of multi-billion-parameter vision-language models MUST NOT be the default path.
The default progression is Grounding DINO for zero-shot reference and Florence-2 Base for
the primary trainable baseline. Larger models, including PaliGemma 2 3B, MAY be explored
only as explicitly optional constrained experiments using memory-saving methods and only
after baseline evidence justifies the cost.

Rationale: a baseline that cannot be run, debugged, or repeated locally is not a valid
project baseline.

### III. Dataset Integrity and Evaluation Isolation
All real-image data MUST be normalized into one shared household-object ontology, one
annotation format, and one documented split policy. A held-out real test set MUST be
created early and MUST remain frozen for the duration of Phase 1. Public datasets MUST be
curated rather than ingested wholesale, and every imported subset MUST record its source,
class mapping, and split assignment. No experiment may use the held-out real test set for
training, hyperparameter tuning, or checkpoint selection.

Rationale: label inconsistency and evaluation leakage can invalidate baseline conclusions
even when model code is correct.

### IV. Reproducible, Measurable Baselines
Every baseline claim MUST be backed by reproducible experiment artifacts. At minimum, each
run MUST record the dataset version or manifest, class ontology, model name, prompt or
tuning configuration, image resolution, batch size, precision mode, seed when applicable,
latency per image, GPU memory use, mAP, precision or recall, per-class AP when available,
and qualitative failure examples. If a run cannot be reproduced or compared fairly, it MUST
NOT be used as the decision-making baseline.

Rationale: the purpose of Phase 1 is not only to train a model but to learn which design
choices actually improve real-world detection under hardware constraints.

### V. Incremental Decision Gates
Work MUST proceed through explicit evidence-based gates:
1. Establish the zero-shot reference baseline.
2. Establish the compact trainable baseline.
3. Review failure taxonomy and hardware cost.
4. Approve any optional stretch experiment only if the earlier gates are complete.

Each gate MUST end with a concise written outcome stating what was tested, what improved,
what failed, and whether the next gate is justified.

Rationale: incremental gates keep the project measurable, prevent scope creep, and align
effort with what the hardware and data can support.

## Current Phase Constraints

The current constitution applies to the Phase 1 baseline effort only.

- In scope:
  - household-object ontology definition
  - real-image dataset curation and normalization
  - split creation and freeze policy
  - Grounding DINO zero-shot evaluation
  - Florence-2 Base baseline training or adaptation
  - optional constrained PaliGemma experiment only after earlier gates complete
  - metric reporting and failure analysis
- Out of scope:
  - NVIDIA Omniverse or Isaac Sim implementation
  - synthetic dataset generation
  - mixed real plus synthetic retraining
  - deployment-camera integration beyond evaluation planning
  - multi-phase roadmap execution beyond documenting later work as deferred

Any artifact that introduces out-of-scope work as current implementation MUST be revised
before approval.

## Workflow and Quality Gates

1. Every feature specification MUST state the Phase 1 objective, target classes, real-data
   sources, split policy, hardware assumptions, success metrics, and explicit out-of-scope
   items.
2. Every implementation plan MUST pass a constitution check covering scope discipline,
   hardware fit, dataset isolation, reproducibility, and gate ordering before design work
   proceeds.
3. Every task list MUST include tasks for dataset normalization, split protection, baseline
   evaluation, metric capture, and failure-analysis write-up where applicable.
4. Reviews MUST reject plans that depend on unapproved simulation work or on compute that
   exceeds the declared workstation budget without a documented waiver.
5. If a required experiment cannot be executed as planned because of hardware, tooling, or
   data limits, the artifact MUST document the limitation and the fallback rather than
   silently omitting the result.

## Governance

This constitution supersedes conflicting guidance in repository planning documents for the
current project phase. Amendments MUST update this file and any affected templates in the
same change. Versioning follows semantic versioning:

- MAJOR for incompatible governance changes or principle removals
- MINOR for new principles, sections, or materially expanded obligations
- PATCH for clarifications that do not change project obligations

Compliance review is mandatory for every new spec, plan, and task set. Reviewers MUST check
that artifacts remain Phase 1 only, fit the local hardware budget, preserve held-out real
test isolation, and define measurable baseline outputs. Any exception MUST be documented
with rationale and explicit approval in the relevant artifact.

**Version**: 1.0.0 | **Ratified**: 2026-04-08 | **Last Amended**: 2026-04-08
