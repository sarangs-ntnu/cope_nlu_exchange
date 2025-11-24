from __future__ import annotations

import argparse
from typing import Dict

from torch.utils.data import DataLoader, random_split

from .data import SyntheticReviewDataset
from .train import TrainConfig, evaluate, evaluate_single, train_joint, train_single_task


def _prepare_loaders(noise: float, seed: int = 0):
    dataset = SyntheticReviewDataset(num_examples=400, aspect_noise=noise, seed=seed)
    train_size = int(0.75 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=None
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    return dataset, train_loader, test_loader


def run_experiment(noise: float, hierarchical: bool, aspect_weight: float) -> Dict[str, float]:
    dataset, train_loader, test_loader = _prepare_loaders(noise=noise)
    config = TrainConfig(hierarchical=hierarchical, aspect_weight=aspect_weight)
    joint_model = train_joint(train_loader, input_dim=len(dataset.vocab), config=config)
    aspect_f1, sentiment_f1 = evaluate(joint_model, test_loader)

    aspect_single = train_single_task(train_loader, len(dataset.vocab), task="aspect")
    sentiment_single = train_single_task(train_loader, len(dataset.vocab), task="sentiment")
    aspect_single_f1 = evaluate_single(aspect_single, test_loader, task="aspect")
    sentiment_single_f1 = evaluate_single(sentiment_single, test_loader, task="sentiment")

    return {
        "hierarchical": float(hierarchical),
        "noise": noise,
        "joint_aspect_f1": aspect_f1,
        "joint_sentiment_f1": sentiment_f1,
        "single_aspect_f1": aspect_single_f1,
        "single_sentiment_f1": sentiment_single_f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dual-head vs single-task comparison")
    parser.add_argument("--noise", type=float, default=0.0, help="Aspect label noise rate (0-1)")
    parser.add_argument("--hierarchical", action="store_true", help="Feed aspect probs to sentiment head")
    parser.add_argument("--aspect-weight", type=float, default=1.0, help="Weight on aspect loss")
    args = parser.parse_args()

    results = run_experiment(noise=args.noise, hierarchical=args.hierarchical, aspect_weight=args.aspect_weight)
    print("Joint multi-task results (hierarchical={}):".format(args.hierarchical))
    print("  Aspect macro F1:    {:.3f}".format(results["joint_aspect_f1"]))
    print("  Sentiment macro F1: {:.3f}".format(results["joint_sentiment_f1"]))
    print("Single-task baselines:")
    print("  Aspect macro F1:    {:.3f}".format(results["single_aspect_f1"]))
    print("  Sentiment macro F1: {:.3f}".format(results["single_sentiment_f1"]))


if __name__ == "__main__":
    main()
