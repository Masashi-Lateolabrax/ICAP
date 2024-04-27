import array
import multiprocessing as mp

from src.optimizer import Hist, TaskGenerator
from src.optimizer.cmaes import base


def _func(individuals: list[base.Individual], task_generator: TaskGenerator, queue: mp.Queue):
    for ind in individuals:
        task = task_generator.generate(ind)
        ind.fitness.values = (task.run(),)
        queue.put(ind)


class ThreadProc(base.ProcInterface):
    def __init__(self, gen: int, thread_id: int, individuals: list[base.Individual], task_generator: TaskGenerator):
        self.gen = gen
        self.thread_id = thread_id
        self.n = len(individuals)
        self.individuals = individuals
        self.queue = mp.Queue(len(individuals))
        self.handle = mp.Process(target=_func, args=(individuals, task_generator, self.queue))
        self.handle.start()

    def finished(self) -> bool:
        return self.queue.qsize() == self.n

    def join(self) -> (int, int):
        self.handle.join()
        for origin in self.individuals:
            result = self.queue.get()
            origin.fitness.values = result.fitness.values
        return self.gen, self.thread_id


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
            max_thread: int = 1,
    ):
        self._base = base.BaseCMAES(dim, population, mu, sigma, centroid, minimalize, max_thread, cmatrix)
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

    def optimize_current_generation(self, env_creator: TaskGenerator, proc=ThreadProc):
        self._current_generation += 1
        self._base.optimize_current_generation(
            self._current_generation, self._generation, env_creator, proc
        )

    def optimize(self, env_creator: TaskGenerator, proc=ThreadProc):
        for gen in range(1, self._generation + 1):
            self._base.optimize_current_generation(
                gen, self._generation, env_creator, proc
            )
        self._current_generation = self._generation
