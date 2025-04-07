import copy
import datetime
import os
import pickle

import numpy as np
from deap.cma import Strategy

from libs import optimizer as opt


class Queue:
    def __init__(
            self,
            scores_avg: float,
            centroid: np.ndarray,
            min_ind: opt.Individual,
            max_ind: opt.Individual,
            sigma: float
    ):
        self.time = datetime.datetime.now()
        self.scores_avg: float = scores_avg
        self.centroid: np.ndarray = centroid.copy()
        self.min_ind: opt.Individual = copy.deepcopy(min_ind)
        self.max_ind: opt.Individual = copy.deepcopy(max_ind)
        self.sigma: float = sigma


class _Logger:
    def __init__(self):
        self.queues: list[Queue] = []
        self.dim: int = -1
        self.population: int = -1
        self.mu: int = -1
        self.min_gen = 0
        self.max_gen = 0
        self.last_cmatrix = np.zeros(0)

    def log(
            self,
            _num_error, avg, min_idx, max_idx,
            individuals: list[opt.Individual],
            strategy: Strategy
    ):
        ## Extract important variables.
        min_ind = individuals[min_idx]
        max_ind = individuals[max_idx]
        min_score = min_ind.fitness.values[0]
        max_score = max_ind.fitness.values[0]

        ## Remove dump data of previous generation for reduce memory size
        if len(self.queues) > 0:
            q = self.queues[-1]
            q.min_ind.dump = None
            q.max_ind.dump = None

        ## Record information of generation
        self.queues.append(Queue(avg, strategy.centroid, min_ind, max_ind, strategy.sigma))
        self.last_cmatrix = strategy.C.copy()

        ## Update `min_gen` and `max_gen`
        recorded_min_score = self.queues[self.min_gen].min_ind.fitness.values[0]
        recorded_max_score = self.queues[self.max_gen].max_ind.fitness.values[0]
        if recorded_min_score > min_score:
            self.min_gen = len(self.queues) - 1
        if recorded_max_score < max_score:
            self.max_gen = len(self.queues) - 1

        ## Update `dim`, `population`, and `mu` if necessary.
        if self.dim < 0:
            self.dim = len(individuals[0])
        if self.population < 0:
            self.population = len(individuals)
        if self.mu < 0:
            self.mu = strategy.mu

    def get_min(self) -> Queue:
        return self.queues[self.min_gen]

    def get_max(self) -> Queue:
        return self.queues[self.max_gen]

    def get_rangking_Nth(self, n: int) -> Queue:
        ranking = sorted(map(lambda x: (x[1].min_score, x[0]), enumerate(self.queues)))[0:5]
        return self.queues[ranking[n][1]]

    def is_empty(self):
        return len(self.queues) == 0


class Logger(opt.Logger):
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self._logger = _Logger()

    def save_tmp(self, gen):
        file_path = os.path.join(self.save_dir, "TMP_LOG.pkl")
        self.save(file_path)

    def save(self, file_name: str):
        file_path = os.path.join(self.save_dir, file_name)
        with open(file_path, "wb") as f:
            pickle.dump(self._logger, f)

    @staticmethod
    def load(file_path: str):
        this = Logger("")
        with open(file_path, "rb") as f:
            this._logger = pickle.load(f)
        return this

    def log(
            self,
            num_error, avg, min_idx, max_idx,
            individuals: list[opt.Individual],
            strategy: Strategy
    ):
        self._logger.log(num_error, avg, min_idx, max_idx, individuals, strategy)

    def get_min(self) -> Queue:
        return self._logger.get_min()

    def get_max(self) -> Queue:
        return self._logger.get_max()

    def get_rangking_Nth(self, n: int) -> Queue:
        return self._logger.get_rangking_Nth(n)

    def is_empty(self):
        return self._logger.is_empty()

    def __iter__(self):
        return iter(self._logger.queues)

    def __getitem__(self, item):
        return self._logger.queues[item]
