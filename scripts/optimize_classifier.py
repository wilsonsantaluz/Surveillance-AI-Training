import argparse
import json
import logging
from pathlib import Path

import optuna

from surveillance_ml.optimization.study import run_optimization


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize the ResNet18 classifier with Optuna")
    parser.add_argument("--config", type=Path, default=Path("configs/optimization.yaml"))
    parser.add_argument("--study-name")
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the pretrained-weight setting from the optimization config",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    study = run_optimization(args.config, args.study_name, args.pretrained)
    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    result: dict[str, object] = {
        "study_name": study.study_name,
        "trial_count": len(study.trials),
        "completed": len(complete),
        "pruned": sum(trial.state == optuna.trial.TrialState.PRUNED for trial in study.trials),
        "failed": sum(trial.state == optuna.trial.TrialState.FAIL for trial in study.trials),
        "best_trial": (
            {
                "number": study.best_trial.number,
                "value": study.best_value,
                "parameters": study.best_params,
            }
            if complete
            else None
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
