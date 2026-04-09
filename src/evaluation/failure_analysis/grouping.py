from __future__ import annotations

from collections import Counter


def group_failures_by_type(failures: list[dict]) -> list[dict]:
    counts = Counter(item["failure_type"] for item in failures)
    return [
        {"failure_type": failure_type, "count": count}
        for failure_type, count in counts.most_common()
    ]

