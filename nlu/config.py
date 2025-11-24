from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Tuple


@dataclass
class SplitConfig:
    """Configuration for dataset splits."""

    train_size: float = 0.7
    dev_size: float = 0.15
    random_seed: int = 1337
    text_column: str = "text"
    label_column: str = "aspect"

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class ModelConfig:
    """Model hyperparameters for the linear classifier."""

    max_features: int = 5000
    ngram_range: Tuple[int, int] = (1, 2)
    c: float = 1.5
    penalty: str = "l2"
    random_seed: int = 1337

    def to_dict(self) -> Dict[str, float]:
        return {
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "c": self.c,
            "penalty": self.penalty,
            "random_seed": self.random_seed,
        }


@dataclass
class RunConfig:
    """High-level configuration for a training or evaluation run."""

    experiment_name: str = "baseline"
    output_dir: Path = Path("models")
    tracker_file: Path = Path("runs/experiments.csv")
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> Dict[str, object]:
        values = {
            "experiment_name": self.experiment_name,
            "output_dir": str(self.output_dir),
            "tracker_file": str(self.tracker_file),
        }
        values.update({f"split.{k}": v for k, v in self.split.to_dict().items()})
        values.update({f"model.{k}": v for k, v in self.model.to_dict().items()})
        return values

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tracker_file.parent.mkdir(parents=True, exist_ok=True)
