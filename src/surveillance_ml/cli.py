"""Command-line entry points for dataset, training, and evaluation workflows."""

import argparse
import logging
from dataclasses import replace
from pathlib import Path

from surveillance_ml.core.config import load_config
from surveillance_ml.data.synthetic import generate_synthetic_dataset
from surveillance_ml.training.train import run_training, run_training_experiment


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
    train.add_argument("--dataset-root", type=Path)
    train.add_argument("--image-size", type=int)
    train.add_argument("--imbalance-ratio", type=float)
    train.add_argument("--output-dir", type=Path)
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--weight-decay", type=float)
    train.add_argument("--seed", type=int)
    train.add_argument("--optimizer", choices=("adam", "adamw", "sgd"))
    train.add_argument("--fine-tuning-strategy", choices=("head", "partial", "full"))
    train.add_argument("--skip-test-evaluation", action="store_true")
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
    dataset = (
        replace(
            config.dataset,
            root=arguments.dataset_root or config.dataset.root,
            image_size=(
                arguments.image_size
                if arguments.image_size is not None
                else config.dataset.image_size
            ),
            imbalance_ratio=(
                arguments.imbalance_ratio
                if arguments.imbalance_ratio is not None
                else config.dataset.imbalance_ratio
            ),
        )
    )
    model = (
        replace(config.model, strategy=arguments.fine_tuning_strategy)
        if arguments.fine_tuning_strategy
        else config.model
    )
    training = replace(
        config.training,
        output_dir=arguments.output_dir or config.training.output_dir,
        epochs=(
            arguments.epochs if arguments.epochs is not None else config.training.epochs
        ),
        batch_size=(
            arguments.batch_size
            if arguments.batch_size is not None
            else config.training.batch_size
        ),
        learning_rate=(
            arguments.learning_rate
            if arguments.learning_rate is not None
            else config.training.learning_rate
        ),
        weight_decay=(
            arguments.weight_decay
            if arguments.weight_decay is not None
            else config.training.weight_decay
        ),
        seed=arguments.seed if arguments.seed is not None else config.training.seed,
        optimizer=arguments.optimizer or config.training.optimizer,
    )
    config = replace(config, dataset=dataset, model=model, training=training)
    checkpoint_path = (
        run_training_experiment(config, evaluate_test=False).checkpoint_path
        if arguments.skip_test_evaluation
        else run_training(config)
    )
    print(f"Best checkpoint: {checkpoint_path}")
