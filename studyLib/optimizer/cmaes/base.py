import abc
import array
import copy
import datetime
import multiprocessing as mp
import numpy
from deap import cma, base
from studyLib.optimizer import EnvInterface, Hist


def default_start_handler(gen, generation, start_time):
    print(f"[{start_time}] start {gen} gen. ({gen}/{generation}={float(gen) / generation * 100.0}%)")


def default_end_handler(population, gen, generation, start_time, fin_time, avg, min_v, max_v, best):
    elapse = float((fin_time - start_time).total_seconds())
    spd = population / elapse
    e = datetime.timedelta(seconds=(generation - gen) * elapse)
    print(
        f"[{fin_time}] finish {gen} gen. speed[ind/s]:{spd}, avg:{avg}, min:{min_v}, max:{max_v}, best:{best}, etr:{e}"
    )


class FitnessMax(base.Fitness):
    weights = (1.0,)

    def __init__(self, values=()):
        super().__init__(values)


class FitnessMin(base.Fitness):
    weights = (-1.0,)

    def __init__(self, values=()):
        super().__init__(values)


class Individual(array.array):
    fitness: base.Fitness = None

    def __new__(cls, fitness: base.Fitness, arr: numpy.ndarray):
        this = super().__new__(cls, "d", arr)
        this.fitness = fitness
        return this


class MaximizeIndividual(Individual):
    def __new__(cls, arr: numpy.ndarray):
        this = super().__new__(cls, FitnessMax((float("nan"),)), arr)
        return this


class MinimalizeIndividual(Individual):
    def __new__(cls, arr: numpy.ndarray):
        this = super().__new__(cls, FitnessMin((float("nan"),)), arr)
        return this


class Proc(metaclass=abc.ABCMeta):
    def __init__(self, env: EnvInterface):
        self.env = env

    def ready(self):
        pass

    def start(self, index: int, queue: mp.Queue, ind: Individual):
        score = self.env.calc(ind)
        queue.put((index, score))


def proc_launcher(index: int, queue: mp.Queue, ind: Individual, proc: Proc):
    proc.start(index, queue, ind)


class BaseCMAES:
    def __init__(
            self,
            dim: int,
            population: int,
            mu: int = -1,
            sigma: float = 0.3,
            minimalize: bool = True,
            max_thread: int = 1
    ):
        self._best_para: array.array = array.array("d", [0.0] * dim)
        self._history: Hist = Hist(minimalize)
        self._start_handler = default_start_handler
        self._end_handler = default_end_handler
        self.max_thread: int = max_thread

        if minimalize:
            self._ind_type = MinimalizeIndividual
        else:
            self._ind_type = MaximizeIndividual

        if mu <= 0:
            mu = int(population * 0.5)

        self._strategy = cma.Strategy(
            centroid=[0 for _i in range(0, dim)],
            sigma=sigma,
            lambda_=population,
            mu=mu,
            weights="equal"
        )

        # self._strategy = cma.StrategyOnePlusLambda(
        #     parent=self._ind_type(numpy.zeros(dim)),
        #     sigma=sigma,
        #     lambda_=population,
        # )

        self._individuals: list[Individual] = self._strategy.generate(self._ind_type)

    def _generate_new_generation(self) -> (float, float, float, array.array, float):
        avg = 0.0
        min_value = float("inf")
        max_value = -float("inf")
        good_para: array.array = None

        for ind in self._individuals:
            if numpy.isnan(ind.fitness.values[0]):
                return None

            avg += ind.fitness.values[0]

            if ind.fitness.values[0] < min_value:
                min_value = ind.fitness.values[0]
                if self._history.is_minimalize():
                    good_para = ind

            if ind.fitness.values[0] > max_value:
                max_value = ind.fitness.values[0]
                if not self._history.is_minimalize():
                    good_para = ind

        avg /= self._strategy.lambda_

        if self._history.add(avg, min_value, max_value):
            self._best_para = copy.deepcopy(good_para)

        self._strategy.update(self._individuals)

        self._individuals: list[Individual] = self._strategy.generate(self._ind_type)

        return avg, min_value, max_value, good_para, self._history.best

    def optimize_current_generation(self, gen: int, generation: int, proc: Proc) -> array.array:
        import time

        start_time = datetime.datetime.now()
        self._start_handler(gen, generation, start_time)

        res = None
        while res is None:
            queue = mp.Queue(self._strategy.lambda_)
            handles = {}
            for i, ind in enumerate(self._individuals):
                if not numpy.isnan(ind.fitness.values[0]):
                    continue

                proc.ready()
                handles[i] = mp.Process(target=proc_launcher, args=(i, queue, ind, proc))
                handles[i].start()

                while len(handles) - queue.qsize() >= self.max_thread:
                    if queue.empty():
                        time.sleep(0.0001)
                        continue
                    while not queue.empty():
                        index, score = queue.get()
                        self._individuals[index].fitness.values = (score,)
                        h = handles.pop(index)
                        h.join()

            for h in handles.values():
                h.join()

            while not queue.empty():
                i, score = queue.get()
                self._individuals[i].fitness.values = (score,)

            res = self._generate_new_generation()

        avg, min_value, max_value, good_para, best = res

        finish_time = datetime.datetime.now()
        self._end_handler(
            self.get_lambda(), gen, generation,
            start_time, finish_time,
            avg, min_value, max_value, best
        )

        return good_para

    def get_ind(self, index: int) -> array.array:
        if index >= self._strategy.lambda_:
            raise "'index' >= self._strategy.lambda_"
        ind = self._individuals[index]
        return ind

    def get_best_para(self) -> array.array:
        return copy.deepcopy(self._best_para)

    def get_best_score(self) -> float:
        return self._history.best

    def get_history(self) -> Hist:
        return copy.deepcopy(self._history)

    def get_lambda(self) -> int:
        return self._strategy.lambda_

    def set_start_handler(self, handler=default_start_handler):
        self._start_handler = handler

    def set_end_handler(self, handler=default_end_handler):
        self._end_handler = handler
