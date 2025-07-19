import enum
from typing import Callable

import numpy as np

from ..config import Settings


class CalculationState(enum.Enum):
    NOT_STARTED = 0
    CALCULATING = 1
    FINISHED = 2
    CORRUPTED = 3


class Individual:
    def __init__(self, input_array, generation: int, settings: Settings):
        self._parameter = np.asarray(input_array)
        self._fitness = float("inf")
        self._calculation_state = CalculationState.NOT_STARTED
        self._generation = generation
        self._settings = settings

    @property
    def as_ndarray(self) -> np.ndarray:
        return self._parameter

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def settings(self) -> Settings:
        return self._settings

    def set_fitness(self, fitness: float):
        self._fitness = fitness

    def get_fitness(self) -> float:
        return self._fitness

    def get_calculation_state(self) -> CalculationState:
        return self._calculation_state

    def set_calculation_state(self, state: CalculationState):
        self._calculation_state = state

    def copy_from(self, other: 'Individual'):
        if not isinstance(other, Individual):
            raise TypeError("Can only copy from another Individual")
        self._parameter[:] = other._parameter[:]
        self._fitness = other._fitness
        self._calculation_state = other._calculation_state
        self._generation = other._generation
        self._settings = other._settings

    def __reduce__(self):
        return (
            Individual,
            (self._parameter, self._generation, self._settings),
            (
                self._fitness,
                self._calculation_state,
            )
        )

    def __setstate__(self, state):
        self._fitness = state[0]
        self._calculation_state = state[1]

    @property
    def is_ready(self) -> bool:
        return self._calculation_state == CalculationState.NOT_STARTED

    @property
    def is_corrupted(self) -> bool:
        return self._calculation_state == CalculationState.CORRUPTED

    @property
    def is_calculating(self) -> bool:
        return self._calculation_state == CalculationState.CALCULATING

    @property
    def is_finished(self) -> bool:
        return self._calculation_state == CalculationState.FINISHED

    @property
    def shape(self):
        return self._parameter.shape


EvaluationFunction = Callable[[Individual], float]
