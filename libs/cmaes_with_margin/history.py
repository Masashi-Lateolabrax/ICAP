import datetime
import os
import pickle

import numpy as np


class Queue:
    def __init__(
            self,
            scores_avg: float,
            min_score: float,
            min_para: np.ndarray,
            max_score: float,
            max_para: np.ndarray,
    ):
        self.time = datetime.datetime.now()
        self.scores_avg: float = scores_avg
        self.min_score: float = min_score
        self.min_para: np.ndarray = min_para.copy()
        self.max_score: float = max_score
        self.max_para: np.ndarray = max_para.copy()


class _Hist:
    def __init__(self):
        self.queues: list[Queue] = []
        self.dim: int = -1
        self.population: int = -1
        self.min_index = 0
        self.max_index = 0

    def log(
            self,
            avg, min_score, min_para, max_score, max_para, individuals: np.ndarray
    ):
        self.queues.append(Queue(avg, min_score, min_para, max_score, max_para))

        if self.queues[self.min_index].min_score > min_score:
            self.min_index = len(self.queues) - 1
        if self.queues[self.max_index].max_score < max_score:
            self.max_index = len(self.queues) - 1

        if self.dim < 0:
            self.dim = individuals.shape[1]
        if self.population < 0:
            self.population = individuals.shape[0]

    def get_min(self) -> Queue:
        return self.queues[self.min_index]

    def get_max(self) -> Queue:
        return self.queues[self.max_index]

    def get_rangking_Nth(self, n: int) -> Queue:
        ranking = sorted(map(lambda x: (x[1].min_score, x[0]), enumerate(self.queues)))[0:5]
        return self.queues[ranking[n][1]]


class Hist:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self._hist = _Hist()

    def save(self, file_name: str):
        file_path = os.path.join(self.save_dir, file_name)
        with open(file_path, "wb") as f:
            pickle.dump(self._hist, f)
        with open(os.path.join(os.path.dirname(file_path), "test.pkl"), "wb") as f:
            pickle.dump(self._hist.queues, f)

    @staticmethod
    def load(file_path: str):
        this = Hist("")
        with open(file_path, "rb") as f:
            this._hist = pickle.load(f)
        return this

    def log(
            self,
            avg, min_score, min_para, max_score, max_para, individuals: np.ndarray
    ):
        self._hist.log(avg, min_score, min_para, max_score, max_para, individuals)

    def get_min(self) -> Queue:
        return self._hist.get_min()

    def get_max(self) -> Queue:
        return self._hist.get_max()

    def get_rangking_Nth(self, n: int) -> Queue:
        return self._hist.get_rangking_Nth(n)
