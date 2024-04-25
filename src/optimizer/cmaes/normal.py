import array
import multiprocessing as mp

from src.optimizer import Hist, EnvCreator
from src.optimizer.cmaes import base


def _func(ind: base.Individual, env_creator: EnvCreator, queue: mp.Queue):
    env = env_creator.create(ind)
    score = env.calc()
    queue.put(score)


class ThreadProc(base.ProcInterface):
    def __init__(self, gen: int, i: int, ind: base.Individual, env_creator: EnvCreator):
        self.gen = gen
        self.i = i
        self.queue = mp.Queue(1)
        self.handle = mp.Process(target=_func, args=(ind, env_creator, self.queue))
        self.handle.start()

    def finished(self) -> bool:
        return self.queue.qsize() > 0

    def join(self) -> (int, int, float):
        self.handle.join()
        return self.gen, self.i, self.queue.get()


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

    def optimize_current_generation(self, gen: int, env_creator: EnvCreator, proc=ThreadProc):
        self._base.optimize_current_generation(gen, self._generation, env_creator, proc)

    def optimize(self, env_creator: EnvCreator, proc=ThreadProc):
        for gen in range(1, self._generation + 1):
            self.optimize_current_generation(gen, env_creator, proc)
