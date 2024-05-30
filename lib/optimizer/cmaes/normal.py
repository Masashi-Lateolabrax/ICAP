import array
import psutil

from lib.optimizer import Hist, TaskGenerator
from lib.optimizer.cmaes import base
from lib.optimizer.processe import MultiThreadProc


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
    ):
        num_cpu = psutil.cpu_count(logical=False)
        if split_tasks <= 0:
            if num_cpu < 2:
                split_tasks = 1
            else:
                split_tasks = num_cpu - 2

        self._base = base.BaseCMAES(dim, population, mu, sigma, centroid, minimalize, split_tasks, cmatrix)
        self._generation = generation
        self._current_generation = 0

    def get_best_para(self) -> array.array:
        return self._base.get_best_para()

    def get_best_score(self) -> float:
        return self.get_best_score()

    def get_history(self) -> Hist:
        return self._base.get_history()

    def set_start_handler(self, handler=base.default_start_handler):
        self._base.set_start_handler(handler)

    def set_end_handler(self, handler=base.default_end_handler):
        self._base.set_end_handler(handler)

    def get_generation(self):
        return self._generation

    def get_current_generation(self):
        return self._current_generation

    def optimize_current_generation(self, env_creator: TaskGenerator, proc=MultiThreadProc):
        self._current_generation += 1
        self._base.optimize_current_generation(
            self._current_generation, self._generation, env_creator, proc
        )

    def optimize(self, env_creator: TaskGenerator, proc=MultiThreadProc):
        for gen in range(1, self._generation + 1):
            self._base.optimize_current_generation(
                gen, self._generation, env_creator, proc
            )
        self._current_generation = self._generation
