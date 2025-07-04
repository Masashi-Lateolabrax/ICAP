import enum
from typing import Callable

import numpy as np


class CalculationState(enum.Enum):
    NOT_STARTED = 0
    SENDING = 1
    CALCULATING = 2
    FINISHED = 3
    CORRUPTED = 4


class Individual(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        obj._fitness = float("inf")
        obj._calculation_state = CalculationState.NOT_STARTED
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._fitness = getattr(obj, '_fitness', None)
        self._calculation_state = getattr(obj, '_calculation_state', CalculationState.NOT_STARTED)

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

    def __reduce__(self):
        return (Individual, (self.view(np.ndarray),), (self._fitness, self._calculation_state))

    def __setstate__(self, state):
        self._fitness = state[0]
        self._calculation_state = state[1]

    @property
    def is_calculating(self) -> bool:
        return self._calculation_state == CalculationState.CALCULATING

    @property
    def is_finished(self) -> bool:
        return self._calculation_state == CalculationState.FINISHED

    @property
    def is_ready(self) -> bool:
        return self._calculation_state == CalculationState.NOT_STARTED

    @property
    def is_sending(self) -> bool:
        return self._calculation_state == CalculationState.SENDING

    @property
    def is_corrupted(self) -> bool:
        return self._calculation_state == CalculationState.CORRUPTED


EvaluationFunction = Callable[[Individual], float]