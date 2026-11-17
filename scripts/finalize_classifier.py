import argparse
from pathlib import Path

from surveillance_ml.optimization.study import finalize_best_trial


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain the best validation-selected Optuna configuration and evaluate test"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/optimization.yaml"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    checkpoint = finalize_best_trial(args.config, args.output_dir)
    print(f"Finalized test-evaluated checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
