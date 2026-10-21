"""Export helpers for similarity analysis."""
from __future__ import annotations

from io import StringIO
import csv


def results_to_csv(results: list[dict]) -> str:
    output = StringIO()
    fieldnames = ["Document A", "Document B", "Similarity (%)"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
    return output.getvalue()
