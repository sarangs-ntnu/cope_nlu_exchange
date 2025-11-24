import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np


def set_seed(seed: int) -> None:
    """Set seeds across supported libraries to guarantee reproducibility."""

    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)
