from __future__ import annotations

from pathlib import Path

from src.data.manifests.models import BenchmarkManifest
from src.synthetic.isaac_script import render_isaac_replicator_script
from src.synthetic.planner import build_synthetic_generation_plan, load_phase2_synthetic_config
from src.utils.paths import read_json


def test_synthetic_plan_targets_lowest_ap_class_from_phase1_summary(repo_root: Path) -> None:
    manifest = BenchmarkManifest.model_validate(
        read_json(repo_root / "tests" / "fixtures" / "phase1_yolo" / "sample_benchmark_manifest.json")
    )
    config = load_phase2_synthetic_config(repo_root / "config" / "phase2.yaml")
    summary = {
        "key_results": {
            "metrics": {
                "per_class_ap50_95": {
                    "mug": 0.48,
                    "book": 0.13,
                }
            }
        },
        "today_issues_and_blockers": ["Book-class AP remains the main quality bottleneck."],
    }

    plan = build_synthetic_generation_plan(manifest, config, phase1_summary=summary)

    assert plan.target_classes == ["book", "mug"]
    assert plan.total_frame_budget == 300
    assert all(pack.provider == "nvidia" for pack in plan.source_references)
    assert "real-only" in plan.heldout_policy


def test_isaac_script_embeds_plan_and_basic_writer(repo_root: Path) -> None:
    manifest = BenchmarkManifest.model_validate(
        read_json(repo_root / "tests" / "fixtures" / "phase1_yolo" / "sample_benchmark_manifest.json")
    )
    config = load_phase2_synthetic_config(repo_root / "config" / "phase2.yaml")
    plan = build_synthetic_generation_plan(manifest, config)

    script = render_isaac_replicator_script(plan)

    assert "import omni.replicator.core as rep" in script
    assert "bounding_box_2d_tight=True" in script
    assert "kitchen_counter_tabletop" in script
    assert "omni.replicator.core" in script

