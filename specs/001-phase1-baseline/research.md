# Phase 0 Research: Phase 1 Baseline Implementation

## Decision 1: Use a uv-managed Python 3.11 project environment

- **Decision**: Target Python 3.11 for the project environment and manage it with `uv`,
even though the current shell reports Python 3.13.5.
- **Rationale**: Phase 1 depends on GPU-enabled ML packages and dataset tooling that are
generally safest when pinned to a conservative, broadly supported Python version. `uv` is
already installed locally, keeps setup lightweight, and supports reproducible dependency
management for a repo that does not yet have an established environment.
- **Alternatives considered**:
  - Use the current system Python 3.13.5 directly: rejected because dependency compatibility
  risk is higher for ML packages and would make the first implementation pass less stable.
  - Use Conda as the default environment manager: rejected because the repo is currently
  lightweight and `uv` is already available on this workstation.

## Decision 2: Implement two required baseline families and gate the stretch model

- **Decision**: Phase 1 MVP will implement Grounding DINO as the required zero-shot
reference baseline and Florence-2 Base as the required compact trainable baseline.
PaliGemma 2 3B remains a documented optional stretch experiment only after the required
baselines and failure review are complete.
- **Rationale**: This sequence satisfies the constitution's hardware and decision-gate
rules. It gives one no-training benchmark and one trainable local benchmark while avoiding
early commitment to a more memory-demanding model path.
- **Alternatives considered**:
  - Florence-2 only: rejected because Phase 1 needs a zero-shot reference point for fair
  comparison.
  - PaliGemma as a required baseline: rejected because it adds memory and tuning complexity
  that is not necessary for the first validated benchmark.

## Decision 3: Extract labeled data in the order COCO -> Open Images -> Objects365

- **Decision**: Start extraction with COCO 2017, add targeted Open Images V7 slices for
missing or weakly covered classes, and use Objects365 only if important ontology gaps
remain after the first two sources.
- **Rationale**: COCO provides the fastest clean starting point with standard splits and a
manageable class set. Open Images provides breadth without forcing a full mirror when
class-filtered extraction is used. Objects365 is valuable for coverage but has more
download and cleanup overhead, so it belongs later in the sequence.
- **Alternatives considered**:
  - Open Images first: rejected because the first implementation pass benefits more from
  COCO's simpler standard benchmark structure.
  - Objects365 first: rejected because it increases extraction overhead before the ontology
  and baseline pipeline are stable.
  - Community-only household datasets: rejected as the primary source because label quality,
  licensing, and ontology consistency vary too much for the initial benchmark.

## Decision 4: Keep the ontology intentionally narrow for the first benchmark cycle

- **Decision**: Target a compact 10-15 class household ontology for the initial benchmark
cycle, using the README's proposed classes as the starting reference set.
- **Rationale**: A smaller ontology reduces normalization effort, improves split quality,
and makes error analysis tractable on local hardware. It also aligns with the Phase 1
goal of a trustworthy baseline rather than a maximally broad detector.
- **Alternatives considered**:
  - Open-vocabulary or large-ontology detection in Phase 1: rejected because it increases
  ambiguity, data-cleaning cost, and evaluation complexity.
  - Extremely small ontology under 5 classes: rejected because it would underrepresent the
  household-use objective and limit baseline learning value.

## Decision 5: Standardize the benchmark and result artifacts as file-based contracts

- **Decision**: Store benchmark definitions and baseline outputs as versioned file artifacts:
a benchmark manifest, baseline run reports, and failure-example galleries.
- **Rationale**: File-based contracts are easy to validate, review, and reproduce in a
single-machine Phase 1 workflow. They also support later scale-up without requiring a
database in the MVP.
- **Alternatives considered**:
  - Notebook-only outputs: rejected because they are harder to validate and compare across
  runs.
  - Database-backed experiment tracking as a requirement: rejected because it adds setup
  overhead before the baseline workflow is proven.

## Decision 6: Use a CLI-first internal workflow

- **Decision**: Expose the Phase 1 pipeline through CLI commands for benchmark preparation,
baseline execution, and summary generation.
- **Rationale**: A CLI-first workflow is the simplest interface for repeatable local runs,
scriptable checks, and future automation. It also fits a repo that currently has no
frontend or service layer.
- **Alternatives considered**:
  - Notebook-first workflow: rejected because it is less structured for repeatable run
  logging and split-protection checks.
  - Web dashboard as a required interface: rejected because it adds non-essential work to a
  Phase 1 baseline effort.

