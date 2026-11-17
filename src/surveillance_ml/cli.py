"""Command-line entry points for the Phase 1 dataset and training workflows."""

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from surveillance_ml.core.config import load_config
from surveillance_ml.data.synthetic import generate_synthetic_dataset
from surveillance_ml.training.train import run_training


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="surveillance-ml")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate-demo", help="Generate labeled synthetic images")
    generate.add_argument("--output-dir", type=Path, default=Path("data/demo"))
    generate.add_argument("--samples-per-class", type=int, default=24)
    generate.add_argument("--image-size", type=int, default=64)
    generate.add_argument("--seed", type=int, default=42)
    train = commands.add_parser("train", help="Train the configured ResNet18 classifier")
    train.add_argument("--config", type=Path, default=Path("configs/classification.yaml"))
    train.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the configured public ImageNet ResNet18 weights setting",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    arguments = _parser().parse_args()
    if arguments.command == "generate-demo":
        path = generate_synthetic_dataset(
            arguments.output_dir,
            samples_per_class=arguments.samples_per_class,
            image_size=arguments.image_size,
            seed=arguments.seed,
        )
        print(f"Generated synthetic dataset: {path}")
        return
    config = load_config(arguments.config)
    if arguments.pretrained is not None:
        config = replace(config, model=replace(config.model, pretrained=arguments.pretrained))
    checkpoint_path = run_training(config)
    print(f"Best checkpoint: {checkpoint_path}")
