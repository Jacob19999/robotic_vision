from __future__ import annotations

import json
from pathlib import Path

from src.cli.summarize_phase1 import summarize_phase1
from src.utils.paths import read_json


def test_summarize_phase1_selects_best_completed_report(repo_root: Path, tmp_path: Path) -> None:
    source_path = repo_root / "tests" / "fixtures" / "phase1_reports" / "sample_summary_inputs.json"
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    payloads = json.loads(source_path.read_text(encoding="utf-8"))
    for index, payload in enumerate(payloads):
        (reports_dir / f"report-{index}.json").write_text(json.dumps(payload), encoding="utf-8")

    output_path = tmp_path / "phase1-summary.json"
    summary = summarize_phase1(reports_dir, output_path)
    saved = read_json(output_path)

    assert summary["recommended_path"] == "florence2"
    assert saved["manifest_id"] == "phase1-sample"
    assert saved["top_failures"][0]["count"] >= 1

