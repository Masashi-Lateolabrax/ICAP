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


class Individual(array.array):
    def __new__(cls, fitness: base.Fitness, arr: numpy.ndarray):
        this = super().__new__(cls, "d", arr)
        this.fitness = fitness
        this.dump = None
        return this

    def __copy__(self):
        new_instance = Individual(self.fitness, self)
        new_instance.dump = self.dump
        return new_instance

    def __deepcopy__(self, memo):
        fitness_copy = copy.deepcopy(self.fitness)
        new_instance = Individual(fitness_copy, self)
        new_instance.dump = copy.deepcopy(self.dump)  # dumpのコピー
        return new_instance


class MaximizeIndividual(Individual):
    def __new__(cls, arr: numpy.ndarray):
        this = super().__new__(cls, FitnessMax((float("nan"),)), arr)
        return this


class MinimalizeIndividual(Individual):
    def __new__(cls, arr: numpy.ndarray):
        this = super().__new__(cls, FitnessMin((float("nan"),)), arr)
        return this
