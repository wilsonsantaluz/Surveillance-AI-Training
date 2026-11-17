import argparse
import json
from dataclasses import replace
from pathlib import Path

from surveillance_ml.core.config import load_config
from surveillance_ml.training.evaluate_checkpoint import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a classifier checkpoint on test data")
    parser.add_argument("--config", type=Path, default=Path("configs/classification.yaml"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/classifier/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.dataset_root is not None:
        config = replace(config, dataset=replace(config.dataset, root=args.dataset_root))
    if args.image_size is not None:
        config = replace(config, dataset=replace(config.dataset, image_size=args.image_size))
    if args.seed is not None:
        config = replace(config, training=replace(config.training, seed=args.seed))
    report = evaluate_checkpoint(config, args.checkpoint, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
