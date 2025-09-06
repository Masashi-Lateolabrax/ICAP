import datetime
import os
import dataclasses
import logging
import pickle
from typing import Self

from .communication import Task


@dataclasses.dataclass(frozen=True)
class OptimizationResult:
    generation: int
    tasks: list[Task]
    avg_fitness: float
    min_fitness: float
    max_fitness: float
    variance: float
    median: float
    timestamp: datetime.datetime

    @classmethod
    def new(cls, generation: int, tasks: list[Task]) -> Self:
        timestamp = datetime.datetime.now(datetime.timezone.utc)

        fitnesses = [task.result for task in tasks]
        avg_fitness = sum(fitnesses) / len(fitnesses)
        min_fitness = min(fitnesses)
        max_fitness = max(fitnesses)
        variance = sum((f - avg_fitness) ** 2 for f in fitnesses) / len(fitnesses)
        median = sorted(fitnesses)[len(fitnesses) // 2]

        return cls(
            generation=generation,
            tasks=tasks,
            avg_fitness=avg_fitness,
            min_fitness=min_fitness,
            max_fitness=max_fitness,
            variance=variance,
            median=median,
            timestamp=timestamp
        )

    def save(self, path: str) -> None:
        if not path.endswith('.pkl'):
            logging.warning(f"File extension is not .pkl. Saving as .pkl.")
            path += '.pkl'
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> Self:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist.")
        if not path.endswith('.pkl'):
            raise ValueError(f"File {path} is not a .pkl file.")
        with open(path, 'rb') as f:
            this = pickle.load(f)
        if not isinstance(this, OptimizationResult):
            raise ValueError(f"File {path} is not a OptimizationResult instance.")
        return this


@dataclasses.dataclass(frozen=True)
class OptimizerResultSet:
    results: dict[int, OptimizationResult]  # key: generation

    @classmethod
    def load_folder(cls, path: str) -> Self:
        results = {}
        for filename in os.listdir(path):
            if not filename.endswith('.pkl'):
                continue
            with open(os.path.join(path, filename), 'rb') as f:
                result = pickle.load(f)
            if not isinstance(result, OptimizationResult):
                logging.warning(f"File {filename} is not a OptimizationResult instance. Skipping.")
                continue
            results[result.generation] = result
        return cls(results=results)

    @classmethod
    def load(cls, path: str) -> Self:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist.")
        if not path.endswith('.pkl'):
            raise ValueError(f"File {path} is not a .pkl file.")
        with open(path, 'rb') as f:
            this = pickle.load(f)
        if not isinstance(this, OptimizerResultSet):
            raise ValueError(f"File {path} is not a OptimizerResultSet instance.")
        return this

    def save(self, path: str) -> None:
        if not path.endswith('.pkl'):
            logging.warning(f"File extension is not .pkl. Saving as .pkl.")
            path += '.pkl'
        with open(path, 'wb') as f:
            pickle.dump(self, f)
