import enum
from typing import Callable, Optional
import time

import numpy as np


class CalculationState(enum.Enum):
    NOT_STARTED = 0
    CALCULATING = 1
    FINISHED = 2
    CORRUPTED = 3


class Individual(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        obj._fitness = float("inf")
        obj._calculation_state = CalculationState.NOT_STARTED
        obj._calculation_start = -1
        obj._calculation_end = -1
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

    def timer_start(self):
        if self._calculation_start != -1:
            raise RuntimeError("Calculation start time already set")
        self._calculation_start = time.time()

    def timer_end(self):
        if self._calculation_start == -1:
            raise RuntimeError("Calculation start time not set")
        if self._calculation_end != -1:
            raise RuntimeError("Calculation end time already set")
        self._calculation_end = time.time()

    def copy_from(self, other: 'Individual'):
        if not isinstance(other, Individual):
            raise TypeError("Can only copy from another Individual")
        self[:] = other[:]
        self._fitness = other._fitness
        self._calculation_state = other._calculation_state
        self._calculation_start = other._calculation_start
        self._calculation_end = other._calculation_end

    def __reduce__(self):
        return (
            Individual,
            (self.to_ndarray(),),
            (
                self._fitness,
                self._calculation_state,
                self._calculation_start,
                self._calculation_end
            )
        )

    def __setstate__(self, state):
        self._fitness = state[0]
        self._calculation_state = state[1]
        self._calculation_start = state[2]
        self._calculation_end = state[3]

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

    def get_elapse(self):
        if self._calculation_start == -1 or self._calculation_end == -1:
            raise RuntimeError("Calculation start or end time not set")
        return self._calculation_end - self._calculation_start


EvaluationFunction = Callable[[Individual], float]
