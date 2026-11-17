"""Validated Optuna search-space definitions."""

from dataclasses import dataclass
from typing import Any

import optuna


@dataclass(frozen=True)
class FloatRange:
    minimum: float
    maximum: float
    log: bool

    @classmethod
    def from_mapping(cls, value: Any, name: str) -> "FloatRange":
        if not isinstance(value, dict):
            raise ValueError(f"search_space.{name} must be a mapping")
        try:
            result = cls(
                minimum=float(value["min"]),
                maximum=float(value["max"]),
                log=bool(value.get("log", False)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid search_space.{name}: {error}") from error
        if result.minimum <= 0 or result.maximum <= result.minimum:
            raise ValueError(f"search_space.{name} requires 0 < min < max")
        return result


@dataclass(frozen=True)
class SearchSpace:
    learning_rate: FloatRange
    weight_decay: FloatRange
    batch_sizes: tuple[int, ...]
    optimizers: tuple[str, ...]
    strategies: tuple[str, ...]
    scheduler_factors: tuple[float, ...]
    scheduler_patiences: tuple[int, ...]

    @classmethod
    def from_mapping(cls, value: Any) -> "SearchSpace":
        if not isinstance(value, dict):
            raise ValueError("search_space must be a mapping")
        try:
            result = cls(
                learning_rate=FloatRange.from_mapping(value["learning_rate"], "learning_rate"),
                weight_decay=FloatRange.from_mapping(value["weight_decay"], "weight_decay"),
                batch_sizes=tuple(int(item) for item in value["batch_size"]["choices"]),
                optimizers=tuple(str(item) for item in value["optimizer"]["choices"]),
                strategies=tuple(
                    str(item) for item in value["fine_tuning_strategy"]["choices"]
                ),
                scheduler_factors=tuple(
                    float(item) for item in value["scheduler_factor"]["choices"]
                ),
                scheduler_patiences=tuple(
                    int(item) for item in value["scheduler_patience"]["choices"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid optimization search space: {error}") from error

        if not result.batch_sizes or any(value < 1 for value in result.batch_sizes):
            raise ValueError("Search batch sizes must be positive and non-empty")
        if not result.optimizers or not set(result.optimizers) <= {"adam", "adamw", "sgd"}:
            raise ValueError("Search optimizers must use supported choices: adam, adamw, sgd")
        if not result.strategies or not set(result.strategies) <= {"head", "partial", "full"}:
            raise ValueError("Search fine-tuning strategies must be head, partial, or full")
        if not result.scheduler_factors or any(
            not 0 < factor < 1 for factor in result.scheduler_factors
        ):
            raise ValueError("Scheduler factors must be between 0 and 1")
        if not result.scheduler_patiences or any(
            patience < 0 for patience in result.scheduler_patiences
        ):
            raise ValueError("Scheduler patience values must be non-negative")
        return result

    def sample(self, trial: optuna.Trial) -> dict[str, Any]:
        """Sample one supported configuration from this search space."""
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate",
                self.learning_rate.minimum,
                self.learning_rate.maximum,
                log=self.learning_rate.log,
            ),
            "weight_decay": trial.suggest_float(
                "weight_decay",
                self.weight_decay.minimum,
                self.weight_decay.maximum,
                log=self.weight_decay.log,
            ),
            "batch_size": trial.suggest_categorical("batch_size", list(self.batch_sizes)),
            "optimizer": trial.suggest_categorical("optimizer", list(self.optimizers)),
            "fine_tuning_strategy": trial.suggest_categorical(
                "fine_tuning_strategy", list(self.strategies)
            ),
            "scheduler_factor": trial.suggest_categorical(
                "scheduler_factor", list(self.scheduler_factors)
            ),
            "scheduler_patience": trial.suggest_categorical(
                "scheduler_patience", list(self.scheduler_patiences)
            ),
        }
