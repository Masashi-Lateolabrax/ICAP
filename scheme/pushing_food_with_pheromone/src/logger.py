import copy
import os.path
import pickle
import re

import numpy as np
from deap.cma import Strategy

from libs.optimizer import Logger as optLogger
from libs.optimizer.individual import Individual

from .settings import Settings


class Logger(optLogger):
    def __init__(self, save_dir):
        self.save_dir = save_dir

        self.logs: list[list[Individual]] = []

        self.last_average = np.zeros(0)
        self.last_cmatrix = np.zeros(0)
        self.best_para = np.zeros(0)

    def log(
            self,
            num_error, avg, min_score, min_para, max_score, max_para, best_para,
            individuals: list[Individual],
            strategy: Strategy
    ):
        self.logs.append(copy.deepcopy(individuals))
        self.last_average = strategy.centroid.copy()
        self.last_cmatrix = strategy.C.copy()
        self.best_para = best_para

    def save_tmp(self, gen):
        file_path = os.path.join(self.save_dir, f"LOG_{gen}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(
                (self.logs, self.last_average, self.last_cmatrix, self.best_para),
                f
            )
        self.logs.clear()

    def save(self):
        self.save_tmp("FINISH")

    @staticmethod
    def load_with_abspath(file_path):
        save_dir = os.path.dirname(file_path)
        this = Logger(save_dir)
        with open(file_path, "rb") as f:
            this.logs, this.last_average, this.last_cmatrix, this.best_para = pickle.load(f)
        return this

    def load(self, file_name):
        file_path = os.path.join(self.save_dir, file_name)
        with open(file_path, "rb") as f:
            self.logs, self.last_average, self.last_cmatrix, self.best_para = pickle.load(f)
        return self


class LogLoader:
    class _LogInfo:
        def __init__(self, save_dir, idx):
            self.name = f"LOG_{idx}.pkl"
            self.num = len(Logger(save_dir).load(self.name).logs)
            self.end_gen = (Settings.Optimization.GENERATION - 1) if idx == "FINISH" else int(idx)
            self.start_gen = -1

        def contain(self, gen):
            return self.start_gen <= gen < self.end_gen

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.log_infos: list[LogLoader._LogInfo] = []

        self._buf_info: LogLoader._LogInfo = None
        self._buf_logger: Logger = None

        for fn in [d for d in os.listdir(save_dir) if os.path.splitext(d)[1] == ".pkl"]:
            m = re.search(r'LOG_(.*?)\.pkl', fn)
            if m:
                print(f"Find: {fn}")
                self.log_infos.append(LogLoader._LogInfo(self.save_dir, m.group(1)))
        self.log_infos.sort(key=lambda x: x.end_gen)

        for i in range(0, len(self.log_infos)):
            self.log_infos[i].start_gen = self.log_infos[i].end_gen - self.log_infos[i - 1].num

        self.length = sum([i.num for i in self.log_infos])

    def get_loader(self, gen) -> Logger | None:
        if gen < 0:
            gen = self.length + gen

        if self._buf_logger is not None and self._buf_info.contain(gen):
            return self._buf_logger

        for i in self.log_infos:
            if i.contain(gen):
                self._buf_info = i
                self._buf_logger = Logger(self.save_dir).load(i.name)
                return self._buf_logger

        return None

    def get_individuals(self, gen) -> list[Individual]:
        if gen < 0:
            gen = self.length + gen
        loader = self.get_loader(gen)
        return loader.logs[gen - self._buf_info.start_gen]

    def get_min_individual(self):
        min_score = float("inf")
        min_ind = None
        gen = -1
        for g in range(0, self.length):
            inds = self.get_individuals(g)
            for i in inds:
                if min_score > i.fitness.values[0]:
                    min_score = i.fitness.values[0]
                    min_ind = i
                    gen = g
        return min_ind, gen

    def get_max_individual(self):
        max_score = -float("inf")
        max_ind = None
        gen = -1
        for g in range(0, self.length):
            inds = self.get_individuals(g)
            for i in inds:
                if max_score < i.fitness.values[0]:
                    max_score = i.fitness.values[0]
                    max_ind = i
                    gen = g
        return max_ind, gen

    def __len__(self):
        return self.length
