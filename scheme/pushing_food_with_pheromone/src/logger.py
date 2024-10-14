import copy
import os.path
import pickle

import numpy as np
from deap.cma import Strategy

from libs.optimizer import Logger as optLogger
from libs.optimizer.individual import Individual


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

    def load(self, gen):
        file_path = os.path.join(self.save_dir, f"LOG_{gen}.pkl")
        with open(file_path, "rb") as f:
            self.logs, self.last_average, self.last_cmatrix, self.best_para = pickle.load(f)
