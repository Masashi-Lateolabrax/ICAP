import os
import dataclasses
import logging
import pickle
from typing import Self

import numpy as np

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

    @classmethod
    def new(cls, generation: int, tasks: list[Task]) -> Self:
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
        )


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



class SavedIndividual:
    def __init__(self, generation, avg_fitness, timestamp, individuals: list[Individual]):
        self.generation = generation
        self.avg_fitness = avg_fitness
        self.timestamp = timestamp
        self.individuals = individuals
        self.num_individuals = len(individuals)

        fitnesses = [ind.get_fitness() for ind in individuals]
        self.best_fitness = max(fitnesses)
        self.worst_fitness = min(fitnesses)

    @staticmethod
    def load(path: str) -> 'SavedIndividual':
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @property
    def best_individual(self):
        return min(self.individuals, key=lambda ind: ind.get_fitness())


@dataclasses.dataclass
class Rec:
    generation: int
    individuals: list[Individual]
    avg_fitness: float
    worst_fitness: float
    best_fitness: float
    variance: float
    median: float

    @staticmethod
    def load(path: str) -> 'Rec':
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save(self, path: str) -> None:
        ext = os.path.splitext(path)[1]
        if ext != '.pkl':
            logging.warning(f"File extension {ext} is not .pkl. Saving as .pkl.")
            path += '.pkl'

        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @property
    def best_individual(self):
        return min(self.individuals, key=lambda ind: ind.get_fitness())


class IndividualRecorder:
    def __init__(self):
        self.recs = {}

    def get_individuals(self, generation: int) -> list[Individual]:
        if generation not in self.recs:
            logging.warning(f"Generation {generation} not found in records.")
            return []
        return self.recs[generation].individuals

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> 'IndividualRecorder':
        with open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def load_from_recs_folder(path: str) -> 'IndividualRecorder':
        import os
        recs = {}
        for filename in os.listdir(path):
            if filename.endswith('.pkl'):
                with open(os.path.join(path, filename), 'rb') as f:
                    rec = pickle.load(f)
                if not isinstance(rec, Rec):
                    logging.warning(f"File {filename} is not a Rec instance. Skipping.")
                    continue
                recs[rec.generation] = rec
        recorder = IndividualRecorder()
        recorder.recs = recs
        return recorder

    def get_best_rec(self) -> Rec:
        return min(self.recs.values(), key=lambda rec: rec.best_fitness)

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, generation: int) -> Rec:
        if generation not in self.recs:
            raise KeyError(f"Generation {generation} not found in records.")
        return self.recs[generation]

    def __iter__(self):
        for generation in sorted(self.recs.keys()):
            yield self.recs[generation]
