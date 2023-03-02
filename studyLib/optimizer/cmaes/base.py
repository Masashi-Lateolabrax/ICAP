import abc
import array
import copy
import datetime
import socket

from deap import cma, base
import numpy
import sys

from studyLib.optimizer import Hist, EnvCreator


def default_start_handler(gen, generation, start_time):
    print(f"[{start_time}] start {gen} gen. ({gen}/{generation}={float(gen) / generation * 100.0}%)")


def default_end_handler(population, gen, generation, start_time, fin_time, num_error, avg, min_v, max_v, best):
    elapse = float((fin_time - start_time).total_seconds())
    spd = population / elapse
    e = datetime.timedelta(seconds=(generation - gen) * elapse)
    print(
        f"[{fin_time}] finish {gen} gen. speed[ind/s]:{spd}, error:{num_error}, avg:{avg}, min:{min_v}, max:{max_v}, best:{best}, etr:{e}"
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
    def __init__(self, gen: int, i: int, ind: Individual, env_creator: EnvCreator):
        raise NotImplemented()

    @abc.abstractmethod
    def finished(self) -> bool:
        raise NotImplemented()

    @abc.abstractmethod
    def join(self) -> (int, int, float):
        raise NotImplemented()


class _OneThreadProc(ProcInterface):
    def __init__(self, gen: int, i: int, ind: array.array, env_creator: EnvCreator):
        self.gen = gen
        self.i = i
        self.score = env_creator.create(ind).calc()

    def finished(self) -> bool:
        return True

    def join(self) -> (int, int, float):
        return self.gen, self.i, self.score


class BaseCMAES:
    def __init__(
            self,
            dim: int,
            population: int,
            mu: int = -1,
            sigma: float = 0.3,
            centroid=None,
            minimalize: bool = True,
            max_thread: int = 1,
            cmatrix=None
    ):
        self._best_para: array.array = array.array("d", [0.0] * dim)
        self._history: Hist = Hist(dim, population, mu)
        self._start_handler = default_start_handler
        self._end_handler = default_end_handler
        self.max_thread: int = max_thread

        self._save_count = 10
        self._save_counter = self._save_count

        if minimalize:
            self._ind_type = _MinimalizeIndividual
            self._best_score = float("inf")
        else:
            self._ind_type = _MaximizeIndividual
            self._best_score = -float("inf")

        if mu <= 0:
            mu = int(population * 0.5)

        if centroid is None:
            centroid = numpy.zeros((dim,))

        if cmatrix is None:
            cmatrix = numpy.identity(dim)

        self._strategy = cma.Strategy(
            centroid=centroid,
            sigma=sigma,
            lambda_=population,
            mu=mu,
            cmatrix=cmatrix
        )

        self._individuals: list[Individual] = self._strategy.generate(self._ind_type)

    def _generate_new_generation(self) -> (int, float, float, float, numpy.array):
        num_error = 0
        avg = 0.0
        min_score = float("inf")
        min_para = None
        max_score = -float("inf")
        max_para = None

        for i, ind in enumerate(self._individuals):
            if numpy.isnan(ind.fitness.values[0]):
                raise "An unknown error occurred."
            elif numpy.isinf(ind.fitness.values[0]):
                num_error += 1
                continue

            avg += ind.fitness.values[0]

            if ind.fitness.values[0] < min_score:
                min_score = ind.fitness.values[0]
                min_para = numpy.array(ind)

            if ind.fitness.values[0] > max_score:
                max_score = ind.fitness.values[0]
                max_para = numpy.array(ind)

        avg /= self._strategy.lambda_ - num_error

        if self._ind_type is _MinimalizeIndividual:
            if self._best_score > min_score:
                self._best_score = min_score
                self._best_para = min_para.copy()
            good_para = min_para.copy()
        else:
            if self._best_score < max_score:
                self._best_score = max_score
                self._best_para = max_para.copy()
            good_para = max_para.copy()

        self._history.add(
            avg,
            self._strategy.centroid,
            min_score,
            min_para,
            max_score,
            max_para,
            self._strategy.sigma,
            self._strategy.C
        )

        self._strategy.update(self._individuals)
        self._individuals: list[Individual] = self._strategy.generate(self._ind_type)

        if self._save_count is not None:
            self._save_counter -= 1
            if self._save_counter <= 0:
                self._history.save()
                self._save_counter = self._save_count

        return num_error, avg, min_score, max_score, good_para

    def optimize_current_generation(
            self, gen: int, generation: int, env_creator: EnvCreator, proc=ProcInterface
    ) -> array.array:
        import time

        start_time = datetime.datetime.now()
        self._start_handler(gen, generation, start_time)

        calculation_finished = False
        handles = []

        try:
            while not calculation_finished:
                calculation_finished = True

                index = 0
                while index < len(handles):
                    if handles[index].finished():
                        p = handles.pop(index)
                        r_gen, r_i, r_score = p.join()
                        if r_gen == gen and not numpy.isnan(r_score):
                            self._individuals[r_i].fitness.values = (r_score,)
                        continue

                    index += 1
                    if self.max_thread <= len(handles) <= index:
                        index = 0
                        time.sleep(0.01)

                for i, ind in enumerate(self._individuals):
                    if not numpy.isnan(ind.fitness.values[0]):
                        continue
                    calculation_finished = False

                    p = proc(gen, i, ind, env_creator)
                    if not p.finished():
                        handles.append(p)
                    else:
                        r_gen, r_i, r_score = p.join()
                        if r_gen == gen and not numpy.isnan(r_score):
                            self._individuals[r_i].fitness.values = (r_score,)

                    if self.max_thread <= len(handles):
                        break

        except KeyboardInterrupt:
            print(f"Interrupt CMAES Optimizing.")
            self._history.save()
            sys.exit()
        except socket.timeout:
            print(f"[CMAES ERROR] Timeout.")
            self._history.save()
            sys.exit()
        except Exception as e:
            print(f"[CMAES ERROR] {e}")
            self._history.save()
            sys.exit()

        num_error, avg, min_value, max_value, good_para = self._generate_new_generation()

        finish_time = datetime.datetime.now()
        self._end_handler(
            self.get_lambda(), gen, generation,
            start_time, finish_time,
            num_error,
            avg, min_value, max_value, self._best_score
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
        return self._best_score

    def get_history(self) -> Hist:
        return copy.deepcopy(self._history)

    def get_lambda(self) -> int:
        return self._strategy.lambda_

    def set_start_handler(self, handler=default_start_handler):
        self._start_handler = handler

    def set_end_handler(self, handler=default_end_handler):
        self._end_handler = handler
