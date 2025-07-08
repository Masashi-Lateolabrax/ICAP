from typing import Optional, List
import logging

import numpy as np
from cmaes import CMA

from icecream import ic
from ..prelude import *


class _IndividualManager:
    def __init__(self):
        self.ready_individuals: List[Individual] = []
        self.assigned_individuals: List[Individual] = []

    @property
    def num_ready_individuals(self) -> int:
        return len(self.ready_individuals)

    def init(self, cmaes: CMA):
        self.ready_individuals = []
        self.assigned_individuals = []
        for _ in range(cmaes.population_size):
            x = cmaes.ask()
            individual = Individual(x)
            self.ready_individuals.append(individual)

    def arrange_individuals(self):
        remaining_individuals = []
        corrupted_count = 0

        for individual in self.assigned_individuals:
            if individual.is_finished or individual.is_calculating:
                remaining_individuals.append(individual)
            else:
                individual.set_calculation_state(CalculationState.NOT_STARTED)
                individual.set_fitness(float("inf"))
                self.ready_individuals.append(individual)
                corrupted_count += 1

        if corrupted_count > 0:
            logging.warning(f"Rearranged {corrupted_count} corrupted individuals back to ready queue")

        self.assigned_individuals = remaining_individuals

    def get_individual(self) -> Optional[Individual]:
        if not self.ready_individuals:
            return None

        individual = self.ready_individuals.pop()
        individual.set_calculation_state(CalculationState.CALCULATING)
        self.assigned_individuals.append(individual)
        return individual

    def all_individuals_finished(self) -> bool:
        if self.ready_individuals:
            return False
        finished_count = sum(1 for ind in self.assigned_individuals if ind.is_finished)
        return finished_count >= len(self.assigned_individuals)


class CMAES:
    def __init__(
            self,
            dimension: int,
            max_generation: int,
            mean: Optional[np.ndarray] = None,
            sigma: float = 1.0,
            population_size: Optional[int] = None
    ):
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(f"dimension must be a positive integer, got {dimension}")

        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")

        if population_size is not None and (not isinstance(population_size, int) or population_size <= 0):
            raise ValueError(f"population_size must be a positive integer, got {population_size}")

        if mean is None:
            mean = np.zeros(dimension)
        elif len(mean) != dimension:
            raise ValueError(f"mean array length {len(mean)} does not match dimension {dimension}")

        self.max_generation = max_generation

        self._optimizer = CMA(
            mean=mean,
            sigma=sigma,
            population_size=population_size,
        )

        self._individual_manager = _IndividualManager()
        self._individual_manager.init(self._optimizer)

    @property
    def individuals(self) -> Individuals:
        return Individuals(self._individual_manager.ready_individuals, self._individual_manager.assigned_individuals)

    @property
    def population_size(self) -> int:
        return self._optimizer.population_size

    @property
    def generation(self) -> int:
        return self._optimizer.generation

    def update(self) -> tuple[bool, list[Individual]]:
        self._individual_manager.arrange_individuals()
        if not ic(self._individual_manager.all_individuals_finished()):
            return False, self._individual_manager.ready_individuals

        individuals = self._individual_manager.assigned_individuals
        solutions = [(i.to_ndarray(), i.get_fitness()) for i in individuals]

        self._optimizer.tell(solutions)
        self._individual_manager.init(self._optimizer)

        return True, individuals

    def get_individuals(self, batch_size: Optional[int] = None) -> Optional[list[Individual]]:
        if batch_size is None or batch_size <= 0:
            return None

        self._individual_manager.arrange_individuals()

        batch = []
        for _ in range(batch_size):
            individual = self._individual_manager.get_individual()
            if individual is not None:
                batch.append(individual)
            else:
                break
        ic(len(self._individual_manager.ready_individuals))

        return batch

    def should_stop(self) -> bool:
        should_stop = self._optimizer.should_stop()

        if self.generation >= self.max_generation:
            logging.warning(f"Reached maximum generation limit: {self.max_generation}")
            should_stop = True

        if should_stop:
            logging.warning("Stopping CMA optimization")

        return should_stop
