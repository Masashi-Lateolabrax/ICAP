from typing import Optional, List, Tuple
import logging
from collections import deque

import numpy as np
from cmaes import CMA

from ..prelude import *


class _IndividualManager:
    def __init__(self):
        self._ready_individuals: deque = deque()
        self._assigned_individuals: List[Individual] = []

    @property
    def num_ready_individuals(self) -> int:
        return len(self._ready_individuals)

    def init(self, cmaes: CMA):
        self._ready_individuals.clear()
        self._assigned_individuals.clear()
        for _ in range(cmaes.population_size):
            x = cmaes.ask()
            individual = Individual(x)
            self._ready_individuals.append(individual)
        logging.debug(f"Initialized {len(self._ready_individuals)} individuals")

    def arrange_individuals(self):
        remaining_individuals = []
        corrupted_count = 0

        for individual in self._assigned_individuals:
            if individual.is_corrupted:
                individual.set_calculation_state(CalculationState.NOT_STARTED)
                individual.set_fitness(float("inf"))
                self._ready_individuals.append(individual)
                corrupted_count += 1
            else:
                remaining_individuals.append(individual)

        if corrupted_count > 0:
            logging.warning(f"Rearranged {corrupted_count} corrupted individuals back to ready queue")

        self._assigned_individuals = remaining_individuals

    def get_individual(self) -> Optional[Individual]:
        if not self._ready_individuals:
            return None

        individual = self._ready_individuals.popleft()
        self._assigned_individuals.append(individual)
        return individual

    def all_individuals_finished(self) -> bool:
        if self._ready_individuals:
            return False
        finished_count = sum(1 for ind in self._assigned_individuals if ind.is_finished)
        return finished_count >= len(self._assigned_individuals)

    def get_solutions(self) -> List[Tuple[np.ndarray, float]]:
        """
        Before calling this method, ensure that all assigned individuals are finished by calling `all_individuals_finished()`.
        """
        solutions = []
        finished_count = 0

        for individual in self._assigned_individuals:
            if individual.is_finished:
                solutions.append((individual.to_ndarray(), individual.get_fitness()))
                finished_count += 1

        logging.debug(f"Retrieved {len(solutions)} solutions ({finished_count} finished)")
        return solutions


class CMAES:
    def __init__(
            self,
            dimension: int,
            mean: Optional[np.ndarray] = None,
            sigma: float = 1.0,
            population_size: Optional[int] = None
    ):
        # Input validation
        if not isinstance(dimension, int) or dimension <= 0:
            logging.error(f"Invalid dimension parameter: {dimension}")
            raise ValueError(f"dimension must be a positive integer, got {dimension}")

        if sigma <= 0:
            logging.error(f"Invalid sigma parameter: {sigma}")
            raise ValueError(f"sigma must be positive, got {sigma}")

        if population_size is not None and (not isinstance(population_size, int) or population_size <= 0):
            logging.error(f"Invalid population_size parameter: {population_size}")
            raise ValueError(f"population_size must be a positive integer, got {population_size}")

        if mean is None:
            mean = np.zeros(dimension)
            logging.debug(f"Using default mean vector of zeros for dimension {dimension}")
        elif len(mean) != dimension:
            logging.error(f"Mean array length {len(mean)} does not match dimension {dimension}")
            raise ValueError(f"mean array length {len(mean)} does not match dimension {dimension}")

        self._optimizer = CMA(
            mean=mean,
            sigma=sigma,
            population_size=population_size,
        )
        logging.info(
            f"CMA optimizer initialized successfully with population_size={self._optimizer.population_size}"
        )

        self._individual_manager = _IndividualManager()
        self._individual_manager.init(self._optimizer)
        logging.debug("Individual manager initialized successfully")

    @property
    def population_size(self) -> int:
        return self._optimizer.population_size

    @property
    def generation(self) -> int:
        return self._optimizer.generation

    @property
    def num_ready_individuals(self) -> Optional[int]:
        return self._individual_manager.num_ready_individuals

    def ready_to_update(self) -> bool:
        return self._individual_manager.all_individuals_finished()

    def update(self) -> List[Tuple[np.ndarray, float]]:
        """
        Before calling this method, ensure that all assigned individuals are finished by calling `ready_to_update()`.
        """
        solutions = self._individual_manager.get_solutions()
        if not solutions:
            logging.warning("No solutions available for update")
            return []

        logging.debug(f"Updating CMA with {len(solutions)} solutions")
        self._optimizer.tell(solutions)
        self._individual_manager.init(self._optimizer)

        fitness_values = [fitness for _, fitness in solutions]
        best_fitness = min(fitness_values)
        avg_fitness = sum(fitness_values) / len(fitness_values)
        logging.info(
            f"Generation {self.generation}: Updated with {len(solutions)} solutions (best: {best_fitness:.6f}, avg: {avg_fitness:.6f})"
        )

        return solutions

    def get_individuals(self, batch_size: Optional[int] = None) -> list[Individual]:
        """Get a batch of individuals for processing.
        
        Args:
            batch_size: Number of individuals to retrieve. If None, returns all available individuals.
            
        Returns:
            List of individuals (may be empty if none available)
        """
        if batch_size is None or batch_size <= 0:
            logging.debug("Batch size is None or non-positive, using default population size")
            return []

        self._individual_manager.arrange_individuals()

        batch = []
        for _ in range(batch_size):
            individual = self._individual_manager.get_individual()
            if individual is not None:
                batch.append(individual)
            else:
                break

        logging.debug(f"Retrieved batch of {len(batch)} individuals")
        return batch

    def should_stop(self) -> bool:
        should_stop = self._optimizer.should_stop()
        if should_stop:
            logging.info(f"CMA optimizer convergence reached at generation {self.generation}")
        return should_stop
