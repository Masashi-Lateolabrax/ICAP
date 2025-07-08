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


class Individuals:
    def __init__(self, ready_individuals: list[Individual], assigned_individuals: list[Individual]):
        self.__ready_individuals = ready_individuals
        self.__assigned_individuals = assigned_individuals

    @property
    def num_finished(self) -> int:
        return len(self.__assigned_individuals)

    @property
    def num_ready(self) -> int:
        return len(self.__ready_individuals)

    def __len__(self):
        return len(self.__ready_individuals) + len(self.__assigned_individuals)

    def __getitem__ref(self, index) -> Optional[Individual]:
        if index < 0 or index >= len(self):
            return None
        if index < len(self.__ready_individuals):
            i = self.__ready_individuals[index]
        else:
            i = self.__assigned_individuals[index - len(self.__ready_individuals)]
        return i

    def __getitem__(self, index: int) -> np.ndarray:
        i = self.__getitem__ref(index)
        if i is None:
            raise IndexError(f"Index {index} out of range for individuals list")
        return np.copy(i)

    def get_fitness(self, index: int) -> float:
        i = self.__getitem__ref(index)
        if i is None:
            raise IndexError(f"Index {index} out of range for individuals list")
        return i.get_fitness()


EvaluationFunction = Callable[[Individual], float]
