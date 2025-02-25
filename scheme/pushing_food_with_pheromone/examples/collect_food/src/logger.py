import dataclasses
import os
import pickle

from deap.cma import Strategy
import numpy as np

import libs.optimizer as opt
from libs.optimizer import Individual


@dataclasses.dataclass
class Queue:
    ave: float
    min_score: float
    min_para: np.ndarray
    loss_fn: np.ndarray
    loss_fr: np.ndarray
    loss: float


class _Logger:
    def __init__(self):
        self.last_centroid = None
        self.last_sigma = None
        self.queues: list[Queue] = []

    def log(self, avg, min_score, min_para, individuals: list[Individual], strategy):
        self.last_centroid = strategy.centroid
        self.last_sigma = strategy.sigma

        loss_element = np.array(
            [[(loss_fn, loss_fr) for (loss_fn, loss_fr) in i.dump] for i in individuals]
        )
        loss = np.sum(loss_element, axis=(1, 2))

        self.queues.append(
            Queue(avg, min_score, min_para, loss_element[:, :, 0], loss_element[:, :, 1], loss)
        )

    def get_loss_elements(self, generation, ind_idx, time):
        loss_fn = self.queues[generation].loss_fn[ind_idx, time]
        loss_fr = self.queues[generation].loss_fr[ind_idx, time]
        return loss_fn, loss_fr

    def get_min(self) -> Queue:
        the_most_min_idx = 0
        the_most_min = self.queues[the_most_min_idx].min_score
        for i, q in enumerate(self.queues):
            if q.min_score < the_most_min:
                the_most_min_idx = i
                the_most_min = q.min_score
        return self.queues[the_most_min_idx]

    def get_rangking_Nth(self, n: int) -> Queue:
        sorted_queues = sorted(self.queues, key=lambda q: q.min_score)
        return sorted_queues[n]


class Logger(opt.Logger):
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self._logger: _Logger = _Logger()

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
        this.save_dir = os.path.dirname(file_path)
        with open(file_path, "rb") as f:
            this._logger = pickle.load(f)
        return this

    def log(
            self,
            num_error, avg, min_score, min_para, max_score, max_para, best_para,
            individuals: list[opt.Individual],
            strategy: Strategy
    ):
        self._logger.log(avg, min_score, min_para, individuals, strategy)

    def get_min(self) -> Queue:
        return self._logger.get_min()

    def get_rangking_Nth(self, n: int) -> Queue:
        return self._logger.get_rangking_Nth(n)
