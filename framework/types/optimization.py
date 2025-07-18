import enum
from typing import Callable, Optional

import numpy as np

from ..prelude import Settings


class CalculationState(enum.Enum):
    NOT_STARTED = 0
    CALCULATING = 1
    FINISHED = 2
    CORRUPTED = 3


class Individual(np.ndarray):
    def __new__(cls, input_array, generation: int, settings: Optional['Settings'] = None):
        obj = np.asarray(input_array).view(cls)
        obj._fitness = float("inf")
        obj._calculation_state = CalculationState.NOT_STARTED
        obj._generation = generation
        obj._settings = settings
        return obj

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def settings(self) -> Optional['Settings']:
        return self._settings

    def __array_finalize__(self, obj):
        if obj is None:
            return
        if not isinstance(obj, Individual):
            raise TypeError("Individual can only be created from another Individual or ndarray")

        self._fitness = obj._fitness
        self._calculation_state = obj._calculation_state
        self._generation = obj._generation
        self._settings = obj._settings

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
        self[:] = other[:]
        self._fitness = other._fitness
        self._calculation_state = other._calculation_state
        self._generation = other._generation
        self._settings = other._settings

    def __reduce__(self):
        return (
            Individual,
            (self.to_ndarray(), self._generation, self._settings),
            (
                self._fitness,
                self._calculation_state,
                self._generation,
                self._settings
            )
        )

    def __setstate__(self, state):
        self._fitness = state[0]
        self._calculation_state = state[1]
        self._generation = state[2]
        self._settings = state[3]

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
    def norm(self):
        return np.linalg.norm(self)

    def to_ndarray(self) -> np.ndarray:
        return self.view(np.ndarray)


EvaluationFunction = Callable[[Individual], float]
