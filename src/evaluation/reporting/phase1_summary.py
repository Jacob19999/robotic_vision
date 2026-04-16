from __future__ import annotations

from datetime import datetime, timezone

from src.evaluation.failure_analysis.grouping import group_failures_by_type


def build_phase1_summary(reports: list[dict]) -> dict:
    if not reports:
        raise ValueError("At least one report is required to build a Phase 1 summary.")

    manifest_ids = {report["manifest_id"] for report in reports}
    if len(manifest_ids) != 1:
        raise ValueError("All reports must reference the same benchmark manifest.")

    completed = [report for report in reports if report["status"] == "completed"]
    if not completed:
        raise ValueError("At least one completed required baseline report is needed.")

    ranked = sorted(completed, key=lambda item: item["metrics"].get("mAP", 0.0), reverse=True)
    grouped_failures = group_failures_by_type(
        [failure for report in reports for failure in report.get("failure_examples", [])]
    )

    return {
        "summary_id": f"phase1-summary-{int(datetime.now(timezone.utc).timestamp())}",
        "manifest_id": reports[0]["manifest_id"],
        "run_ids": [report["run_id"] for report in reports],
        "summary_metrics": {report["model_family"]: report["metrics"] for report in completed},
        "top_failures": grouped_failures,
        "recommended_path": ranked[0]["model_family"],
        "blocked_items": [
            {
                "run_id": report["run_id"],
                "model_family": report["model_family"],
                "status": report["status"],
                "notes": report.get("notes", ""),
            }
            for report in reports
            if report["status"] != "completed"
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
