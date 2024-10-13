import datetime
import os

import numpy as np
from deap.cma import Strategy

from .cmaes.logger import Logger
from .individual import Individual


class Hist(Logger):
    class Queue:
        def __init__(
                self,
                scores_avg: float,
                centroid: np.ndarray,
                min_score: float,
                min_para: np.ndarray,
                max_score: float,
                max_para: np.ndarray,
                sigma: float
        ):
            self.time = datetime.datetime.now()
            self.scores_avg: float = scores_avg
            self.centroid: np.ndarray = centroid.copy()
            self.min_score: float = min_score
            self.min_para: np.ndarray = min_para.copy()
            self.max_score: float = max_score
            self.max_para: np.ndarray = max_para.copy()
            self.sigma: float = sigma

    def __init__(self, save_dir):
        self.save_dir = save_dir

        self.queues: list[Hist.Queue] = []
        self.dim: int = -1
        self.population: int = -1
        self.mu: int = -1
        self.min_index = 0
        self.max_index = 0
        self.last_cmatrix = np.zeros(0)

    def log(
            self,
            num_error, avg, min_score, min_para, max_score, max_para, best_para,
            individuals: list[Individual],
            strategy: Strategy
    ):
        self.queues.append(Hist.Queue(avg, strategy.centroid, min_score, min_para, max_score, max_para, strategy.sigma))
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

    def save_tmp(self, gen):
        file_path = os.path.join(self.save_dir, "TMP_LOG.tmp.npz")
        self.save(file_path)

    def save(self, file_name: str):
        file_path = os.path.join(self.save_dir, file_name)

        meta = np.array([self.dim, self.population, self.mu])

        time: list[str] = []
        centroids = np.zeros((len(self.queues), self.dim))
        min_para = np.zeros((len(self.queues), self.dim))
        max_para = np.zeros((len(self.queues), self.dim))
        score = np.zeros((len(self.queues), 3))
        sigmas = np.zeros((len(self.queues),))
        for i, q in enumerate(self.queues):
            time.append(q.time.strftime("%Y-%m-%d %H:%M:%S.%f"))
            centroids[i] = q.centroid
            min_para[i] = q.min_para
            max_para[i] = q.max_para
            score[i] = np.array([q.scores_avg, q.min_score, q.max_score])
            sigmas[i] = q.sigma

        np.savez(
            file_path,
            meta=meta,
            time=time,
            centroids=centroids,
            min_para=min_para,
            max_para=max_para,
            score=score,
            sigmas=sigmas,
            min_index=self.min_index,
            max_index=self.max_index,
            last_cmatrix=self.last_cmatrix,
        )

    @staticmethod
    def load(file_path: str):
        npz = np.load(file_path)

        meta = npz["meta"]
        this = Hist(dim=meta[0], population=meta[1], mu=meta[2])
        this.min_index = npz["min_index"]
        this.max_index = npz["max_index"]
        this.last_cmatrix = npz["last_cmatrix"]

        time = npz["time"]
        centroids = npz["centroids"]
        min_para = npz["min_para"]
        max_para = npz["max_para"]
        score = npz["score"]
        sigmas = npz["sigmas"]
        for t, centroid, min_p, max_p, s, sigma_ in zip(time, centroids, min_para, max_para, score, sigmas):
            q = Hist.Queue(s[0], centroid, s[1], min_p, s[2], max_p, sigma_)
            q.time = datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S.%f")
            this.queues.append(q)

        this.save_dir = os.path.dirname(file_path)

        return this

    def get_min(self) -> Queue:
        return self.queues[self.min_index]

    def get_max(self) -> Queue:
        return self.queues[self.max_index]

    def get_rangking_Nth(self, n: int) -> Queue:
        ranking = sorted(map(lambda x: (x[1].min_score, x[0]), enumerate(self.queues)))[0:5]
        return self.queues[ranking[n][1]]
