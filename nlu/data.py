from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import SplitConfig
from .utils import ensure_dir, set_seed


class Splitter:
    """Create reproducible train/dev/test splits."""

    def __init__(self, config: SplitConfig):
        self.config = config

    def load(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        missing_cols = {self.config.text_column, self.config.label_column} - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        return df

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        set_seed(self.config.random_seed)
        remaining = 1.0 - self.config.train_size
        if remaining <= 0:
            raise ValueError("train_size must be < 1.0 to leave room for dev/test")

        dev_ratio = self.config.dev_size / remaining
        train_df, interim_df = train_test_split(
            df,
            test_size=remaining,
            random_state=self.config.random_seed,
            stratify=df[self.config.label_column],
        )
        dev_df, test_df = train_test_split(
            interim_df,
            test_size=1 - dev_ratio,
            random_state=self.config.random_seed,
            stratify=interim_df[self.config.label_column],
        )
        return train_df.reset_index(drop=True), dev_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def save_splits(self, train_df: pd.DataFrame, dev_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> None:
        ensure_dir(output_dir)
        train_df.to_csv(output_dir / "train.csv", index=False)
        dev_df.to_csv(output_dir / "dev.csv", index=False)
        test_df.to_csv(output_dir / "test.csv", index=False)
