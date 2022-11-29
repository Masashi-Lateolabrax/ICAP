import abc
import array
import copy
import datetime
from deap import cma, base
import numpy
import sys

from studyLib.optimizer import Hist, EnvCreator


def default_start_handler(gen, generation, start_time):
    print(f"[{start_time}] start {gen} gen. ({gen}/{generation}={float(gen) / generation * 100.0}%)")


def default_end_handler(population, gen, generation, start_time, fin_time, avg, min_v, max_v, best):
    elapse = float((fin_time - start_time).total_seconds())
    spd = population / elapse
    e = datetime.timedelta(seconds=(generation - gen) * elapse)
    print(
        f"[{fin_time}] finish {gen} gen. speed[ind/s]:{spd}, avg:{avg}, min:{min_v}, max:{max_v}, best:{best}, etr:{e}"
    )


class Logger(object):
    class LogQueue:
        def __init__(self, centroid, c):
            self.centroid = centroid
            self.C = c

    def __init__(self, dim: int = 0, population: int = 0, mu: int = 0):
        self.dim = dim
        self.population = population
        self.mu = mu
        self.queue: list[Logger.LogQueue] = []

    def save(self):
        data = numpy.zeros((len(self.queue), self.dim + self.dim * self.dim))
        for i, q in enumerate(self.queue):
            data[i] = numpy.concatenate([q.centroid, q.C.ravel()])
        numpy.savez(
            "./CMAES_LOG.log", meta=[self.dim, self.population, self.mu], data=data
        )

    def load(self):
        npz_file = numpy.load("./CMAES_LOG.log.npz")
        meta = npz_file["meta"]
        data = npz_file["data"]
        self.dim = meta[0]
        self.population = meta[1]
        self.mu = meta[2]
        for i in range(0, data.shape[0]):
            self.queue.append(Logger.LogQueue(
                numpy.array(data[i][0:self.dim]),
                numpy.array(data[i][self.dim:self.dim + self.dim * self.dim]).reshape((self.dim, self.dim)),
            ))

    def add_log(self, centroid: numpy.ndarray, c: numpy.ndarray):
        self.queue.append(Logger.LogQueue(centroid.copy(), c.copy()))


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
        if this.fitness is None:
            this.fitness = fitness
        return this


class _MaximizeIndividual(Individual):
    def __new__(cls, arr: numpy.ndarray):
        this = super().__new__(cls, FitnessMax((float("nan"),)), arr)
        return this


class _MinimalizeIndividual(Individual):
    def __new__(cls, arr: numpy.ndarray):
        this = super().__new__(cls, FitnessMin((float("nan"),)), arr)
        return this


class ProcInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, i: int, ind: Individual, env_creator: EnvCreator):
        raise NotImplemented()

    @abc.abstractmethod
    def finished(self) -> bool:
        raise NotImplemented()

    @abc.abstractmethod
    def join(self) -> float:
        raise NotImplemented()


class _OneThreadProc(ProcInterface):
    def __init__(self, ind: array.array, env_creator: EnvCreator):
        self.score = env_creator.create().calc(ind)

    def finished(self) -> bool:
        return True

    def join(self) -> float:
        return self.score


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
        self.logger = Logger(dim, population, mu)

        if minimalize:
            self._ind_type = _MinimalizeIndividual
        else:
            self._ind_type = _MaximizeIndividual

        if mu <= 0:
            mu = int(population * 0.5)

        self._strategy = cma.Strategy(
            centroid=[0 for _i in range(0, dim)],
            sigma=sigma,
            lambda_=population,
            mu=mu,
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

        for i, ind in enumerate(self._individuals):
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

        self.logger.add_log(self._strategy.centroid, self._strategy.C)

        return avg, min_value, max_value, good_para, self._history.best

    def optimize_current_generation(
            self, gen: int, generation: int, env_creator: EnvCreator, proc=ProcInterface
    ) -> array.array:
        import time

        start_time = datetime.datetime.now()
        self._start_handler(gen, generation, start_time)

        recovery_mode = False
        res = None
        while res is None:
            handles = {}
            for i, ind in enumerate(self._individuals):
                if not numpy.isnan(ind.fitness.values[0]):
                    continue
                elif recovery_mode:
                    print(f"Retry No.{i}.")

                try:
                    handles[i] = proc(i, ind, env_creator)
                except KeyboardInterrupt:
                    print(f"Interrupt CMAES Optimizing.")
                    self.logger.save()
                    sys.exit()
                except Exception as e:
                    print(f"[CMAES ERROR] {e}")
                    sys.exit()

                while len(handles) >= self.max_thread:
                    remove_list = []
                    for key in handles.keys():
                        if handles[key].finished():
                            remove_list.append(key)
                    for key in remove_list:
                        p = handles.pop(key)
                        self._individuals[key].fitness.values = (p.join(),)
                    time.sleep(0.0001)

            for key, p in handles.items():
                self._individuals[key].fitness.values = (p.join(),)

            res = self._generate_new_generation()
            recovery_mode = res is None

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
