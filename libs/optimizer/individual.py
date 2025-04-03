import copy

from deap import base
import array
import numpy


class FitnessMax(base.Fitness):
    weights = (1.0,)

    def __init__(self, values=()):
        super().__init__(values)


class FitnessMin(base.Fitness):
    weights = (-1.0,)

    def __init__(self, values=()):
        super().__init__(values)


class Individual(numpy.ndarray):
    def __new__(cls, fitness, arr: numpy.ndarray):
        this = numpy.asarray(arr).view(cls)
        this.fitness = fitness
        this.dump = None
        return this

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.fitness = getattr(obj, 'fitness', None)
        self.dump = getattr(obj, 'dump', None)

    def __copy__(self):
        new_instance = Individual(self.fitness, self)
        new_instance.dump = self.dump
        return new_instance

    def __deepcopy__(self, memo):
        fitness_copy = copy.deepcopy(self.fitness)
        new_instance = Individual(fitness_copy, self)
        new_instance.dump = copy.deepcopy(self.dump)
        return new_instance

    def __reduce__(self):
        return (Individual, (self.fitness, self.view(numpy.ndarray)), (self.fitness, self.dump))

    def __setstate__(self, state):
        self.fitness = state[0]
        self.dump = state[1]


class MaximizeIndividual(Individual):
    def __new__(cls, arr: numpy.ndarray):
        this = super().__new__(cls, FitnessMax((float("nan"),)), arr)
        return this

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)

    def __reduce__(self):
        state = super().__reduce__()
        return (MaximizeIndividual, (self.view(numpy.ndarray),), state[2])

    def __setstate__(self, state):
        super().__setstate__(state)


class MinimalizeIndividual(Individual):
    def __new__(cls, arr: numpy.ndarray):
        this = super().__new__(cls, FitnessMin((float("nan"),)), arr)
        return this

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)

    def __reduce__(self):
        state = super().__reduce__()
        return (MinimalizeIndividual, (self.view(numpy.ndarray),), state[2])

    def __setstate__(self, state):
        super().__setstate__(state)
