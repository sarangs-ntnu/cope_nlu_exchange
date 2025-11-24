import csv
from pathlib import Path
from typing import Dict

from .utils import ensure_dir, timestamp


class CsvTracker:
    """Simple experiment tracker writing configurations and metrics to CSV."""

    def __init__(self, path: Path):
        self.path = path
        ensure_dir(self.path.parent)

    def log(self, config: Dict[str, object], metrics: Dict[str, object]) -> None:
        flat_metrics = {f"metric.{k}": v for k, v in metrics.items() if not isinstance(v, dict)}
        row = {"run_id": timestamp(), **config, **flat_metrics}
        self._append(row)

    def _append(self, row: Dict[str, object]) -> None:
        file_exists = self.path.exists()
        with self.path.open("a", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
