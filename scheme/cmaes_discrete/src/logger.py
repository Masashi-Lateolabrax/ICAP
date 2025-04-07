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
            min_score: float,
            min_ind: opt.Individual,
            max_score: float,
            max_ind: opt.Individual,
            sigma: float
    ):
        self.time = datetime.datetime.now()
        self.scores_avg: float = scores_avg
        self.centroid: np.ndarray = centroid.copy()
        self.min_score: float = min_score
        self.min_para: opt.Individual = copy.copy(min_ind)
        self.max_score: float = max_score
        self.max_para: opt.Individual = copy.copy(max_ind)
        self.sigma: float = sigma


class _Logger:
    def __init__(self):
        self.queues: list[Queue] = []
        self.dim: int = -1
        self.population: int = -1
        self.mu: int = -1
        self.min_index = 0
        self.max_index = 0
        self.last_cmatrix = np.zeros(0)

    def log(
            self,
            _num_error, avg, min_score, min_idx, max_score, max_idx,
            individuals: list[opt.Individual],
            strategy: Strategy
    ):
        min_ind = individuals[min_idx]
        max_ind = individuals[max_idx]
        self.queues.append(Queue(avg, strategy.centroid, min_score, min_ind, max_score, max_ind, strategy.sigma))
        self.last_cmatrix = strategy.C.copy()

        if self.queues[self.min_index].min_score > min_score:
            self.min_index = len(self.queues) - 1
        if self.queues[self.max_index].max_score < max_score:
            self.max_index = len(self.queues) - 1

        if self.dim < 0:
            self.dim = len(individuals[0])
        if self.population < 0:
            self.population = len(individuals)
        if self.mu < 0:
            self.mu = strategy.mu

    def get_min(self) -> Queue:
        return self.queues[self.min_index]

    def get_max(self) -> Queue:
        return self.queues[self.max_index]

    def get_rangking_Nth(self, n: int) -> Queue:
        ranking = sorted(map(lambda x: (x[1].min_score, x[0]), enumerate(self.queues)))[0:5]
        return self.queues[ranking[n][1]]

    def is_empty(self):
        return len(self.queues) == 0


class Logger(opt.Logger):
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self._hist = _Logger()

    def save_tmp(self, gen):
        file_path = os.path.join(self.save_dir, "TMP_LOG.pkl")
        self.save(file_path)

    def save(self, file_name: str):
        file_path = os.path.join(self.save_dir, file_name)
        with open(file_path, "wb") as f:
            pickle.dump(self._hist, f)

    @staticmethod
    def load(file_path: str):
        this = Logger("")
        with open(file_path, "rb") as f:
            this._hist = pickle.load(f)
        return this

    def log(
            self,
            num_error, avg, min_score, min_idx, max_score, max_idx,
            individuals: list[opt.Individual],
            strategy: Strategy
    ):
        self._hist.log(num_error, avg, min_score, min_idx, max_score, max_idx, individuals, strategy)

    def get_min(self) -> Queue:
        return self._hist.get_min()

    def get_max(self) -> Queue:
        return self._hist.get_max()

    def get_rangking_Nth(self, n: int) -> Queue:
        return self._hist.get_rangking_Nth(n)

    def is_empty(self):
        return self._hist.is_empty()
