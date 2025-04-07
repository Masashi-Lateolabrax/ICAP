import copy
import datetime
import socket
import time

from deap import cma
import numpy
import sys

from ..task_interface import TaskGenerator
from ..individual import MinimalizeIndividual, MaximizeIndividual, Individual
from ..processe import ProcInterface

from .logger import Logger


def default_start_handler(gen, generation, start_time):
    print(f"[{start_time}] start {gen} gen. ({gen}/{generation}={float(gen) / generation * 100.0}%)")


def default_end_handler(population, gen, generation, start_time, fin_time, num_error, avg, min_v, max_v, best):
    elapse = float((fin_time - start_time).total_seconds())
    spd = population / elapse
    e = datetime.timedelta(seconds=(generation - gen) * elapse)
    print(
        f"[{fin_time}] finish {gen} gen. speed[ind/s]:{spd}, error:{num_error}, avg:{avg}, min:{min_v}, max:{max_v}, best:{best}, etr:{e}"
    )


class BaseCMAES:
    def __init__(
            self,
            dim: int,
            population: int,
            mu: int = -1,
            sigma: float = 0.3,
            centroid=None,
            minimalize: bool = True,
            split_tasks: int = 1,
            cmatrix=None,
            logger: Logger = None
    ):
        self._best_para: numpy.ndarray = numpy.zeros(dim)
        self._start_handler = default_start_handler
        self._end_handler = default_end_handler
        self._split_tasks: int = split_tasks
        self._current_generation = 0

        self.logger = logger

        self._save_count = 10
        self._save_counter = self._save_count

        if minimalize:
            self._ind_type = MinimalizeIndividual
            self._best_score = float("inf")
        else:
            self._ind_type = MaximizeIndividual
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

    def _check_individuals(self) -> (int, float, (numpy.array, float), (numpy.array, float)):
        num_error = 0
        avg = 0.0
        min_score = float("inf")
        min_idx = None
        max_score = -float("inf")
        max_idx = None

        for i, ind in enumerate(self._individuals):
            if numpy.isnan(ind.fitness.values[0]):
                raise "An unknown error occurred."
            elif numpy.isinf(ind.fitness.values[0]):
                num_error += 1
                continue

            avg += ind.fitness.values[0]

            if ind.fitness.values[0] < min_score:
                min_score = ind.fitness.values[0]
                min_idx = i

            if ind.fitness.values[0] > max_score:
                max_score = ind.fitness.values[0]
                max_idx = i

        avg /= self._strategy.lambda_ - num_error

        if self._ind_type is MinimalizeIndividual:
            if self._best_score > min_score:
                self._best_score = min_score
                self._best_para = self._individuals[min_idx].view().copy()
        else:
            if self._best_score < max_score:
                self._best_score = max_score
                self._best_para = self._individuals[max_idx].view().copy()

        return num_error, avg, (min_score, min_idx), (max_score, max_idx), self._best_para

    def _divide_tasks(self) -> list[list[Individual]]:
        num_task = int(len(self._individuals) / self._split_tasks)
        tasks = [list(self._individuals[s * num_task:(s + 1) * num_task]) for s in range(self._split_tasks)]
        for thread_id in range(len(self._individuals) % self._split_tasks):
            tasks[thread_id].append(self._individuals[-1 - thread_id])
        return tasks

    def _optimize(self, task_generator: TaskGenerator, proc=ProcInterface):
        tasks = self._divide_tasks()
        try:
            handles = []
            for i, ts in enumerate(tasks):
                handles.append(proc(self._current_generation, i, ts, task_generator))

            while len(handles) > 0:
                index = 0
                while index < len(handles):
                    if handles[index].finished():
                        h = handles.pop(index)
                        h.join()
                        continue
                    index += 1
                time.sleep(0.01)

        except KeyboardInterrupt:
            print(f"Interrupt CMAES Optimizing.")
            return False
        except socket.timeout:
            print(f"[CMAES ERROR] Timeout.")
            return False
        except Exception as e:
            print(f"[CMAES ERROR] {e}")
            return False
        return True

    def optimize_current_generation(
            self, generation: int, task_generator: TaskGenerator, proc=ProcInterface
    ) -> tuple[int, float, float, float, numpy.ndarray]:

        start_time = datetime.datetime.now()
        self._start_handler(self._current_generation, generation, start_time)

        if not self._optimize(task_generator, proc):
            sys.exit()

        num_error, avg, (min_score, min_idx), (max_score, max_idx), best_para = self.update()

        self.log(
            num_error, avg, min_idx, max_idx
        )

        finish_time = datetime.datetime.now()
        self._end_handler(
            self.get_lambda(), self._current_generation, generation,
            start_time, finish_time,
            num_error,
            avg, min_score, max_score, self._best_score
        )

        return num_error, avg, min_score, max_score, best_para

    def log(self, num_error, avg, min_idx, max_idx):
        if self.logger is not None:
            self.logger.log(
                num_error, avg, min_idx, max_idx,
                self._individuals,
                self._strategy
            )

            if self._save_count is not None:
                self._save_counter -= 1
                if self._save_counter <= 0:
                    self.logger.save_tmp(self._current_generation)
                    self._save_counter = self._save_count

    def update(self):
        num_error, avg, (min_score, min_idx), (max_score, max_idx), best_para = self._check_individuals()

        self._strategy.update(self._individuals)
        self._individuals: list[Individual] = self._strategy.generate(self._ind_type)

        self._current_generation += 1

        return num_error, avg, (min_score, min_idx), (max_score, max_idx), best_para

    def get_ind(self, index: int) -> Individual:
        if index >= self._strategy.lambda_:
            raise "'index' >= self._strategy.lambda_"
        ind = self._individuals[index]
        return ind

    def get_best_para(self) -> numpy.ndarray:
        return copy.deepcopy(self._best_para)

    def get_best_score(self) -> float:
        return self._best_score

    def get_lambda(self) -> int:
        return self._strategy.lambda_

    def set_start_handler(self, handler=default_start_handler):
        self._start_handler = handler

    def set_end_handler(self, handler=default_end_handler):
        self._end_handler = handler

    def get_current_generation(self):
        return self._current_generation
