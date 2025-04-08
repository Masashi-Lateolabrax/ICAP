import numpy
import psutil

from ..task_interface import TaskGenerator
from ..processe import MultiThreadProc
from .logger import Logger
from . import base


class CMAES:
    def __init__(
            self,
            dim: int,
            generation: int,
            population: int,
            mu: int = -1,
            sigma: float = 0.3,
            centroid=None,
            cmatrix=None,
            minimalize: bool = True,
            split_tasks: int = 0,
            logger: Logger = None
    ):
        num_cpu = psutil.cpu_count(logical=False)
        if split_tasks <= 0:
            if num_cpu < 2:
                split_tasks = 1
            else:
                split_tasks = num_cpu - 2

        self._base = base.BaseCMAES(dim, population, mu, sigma, centroid, minimalize, split_tasks, cmatrix, logger)
        self._generation = generation

    def get_lambda(self):
        return self._base.get_lambda()

    def get_individual(self, index: int) -> base.Individual:
        return self._base.get_ind(index)

    def get_best_para(self) -> numpy.ndarray:
        return self._base.get_best_para()

    def get_best_score(self) -> float:
        return self.get_best_score()

    def set_start_handler(self, handler=base.default_start_handler):
        self._base.set_start_handler(handler)

    def set_end_handler(self, handler=base.default_end_handler):
        self._base.set_end_handler(handler)

    def get_generation(self):
        return self._generation

    def get_current_generation(self):
        return self._base.get_current_generation()

    def log(self, num_error, avg, min_idx, max_idx):
        return self._base.log(num_error, avg, min_idx, max_idx)

    def update(self):
        return self._base.update()

    def optimize_current_generation(
            self, env_creator: TaskGenerator, proc=MultiThreadProc
    ) -> tuple[int, float, float, float, numpy.ndarray]:
        num_err, ave_score, min_score, max_score, best_para = self._base.optimize_current_generation(
            self._generation, env_creator, proc
        )
        return num_err, ave_score, min_score, max_score, best_para

    def optimize(self, env_creator: TaskGenerator, proc=MultiThreadProc):
        for _ in range(1, self._generation + 1):
            self._base.optimize_current_generation(
                self._generation, env_creator, proc
            )
