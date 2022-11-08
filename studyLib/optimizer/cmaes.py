import math
import datetime
import numpy
import copy

from deap import cma, creator, base

from studyLib.optimizer.history import Hist
from studyLib.optimizer.env_interface import EnvInterface


def default_start_handler(gen, generation, start_time):
    print(f"[{start_time}] start {gen} gen. ({gen}/{generation}={float(gen) / generation * 100.0}%)")


def default_end_handler(population, gen, generation, start_time, fin_time, avg, min_v, max_v, best):
    elapse = float((fin_time - start_time).total_seconds())
    spd = population / elapse
    e = datetime.timedelta(seconds=(generation - gen) * elapse)
    print(
        f"[{fin_time}] finish {gen} gen. speed[ind/s]:{spd}, avg:{avg}, min:{min_v}, max:{max_v}, best:{best}, etr:{e}"
    )


class _BaseCMAES:
    def __init__(self, dim: int, population: int, sigma=0.3, minimalize=True):
        self._best_para = numpy.zeros(0)
        self._history = Hist(minimalize)

        if minimalize:
            creator.create("Fitness", base.Fitness, weights=(-1.0,))
        else:
            creator.create("Fitness", base.Fitness, weights=(1.0,))
        creator.create("Individual", numpy.ndarray, fitness=creator.Fitness)

        self._strategy = cma.Strategy(
            centroid=[0 for _i in range(0, dim)],
            sigma=sigma,
            lambda_=population
        )
        self._individuals = self._strategy.generate(creator.Individual)
        for ind in self._individuals:
            ind.fitness.values = (float("nan"),)

    def generate(self):
        avg = 0.0
        min_value = float("inf")
        max_value = -float("inf")
        min_para = None
        max_para = None

        for ind in self._individuals:
            if math.isnan(ind.fitness.values[0]):
                return False

            avg += ind.fitness.values[0]

            if ind.fitness.values[0] < min_value:
                min_value = ind.fitness.values[0]
                min_para = ind
            if ind.fitness.values[0] > max_value:
                max_value = ind.fitness.values[0]
                max_para = ind

        avg /= self._strategy.lambda_

        if self._history.add(avg, min_value, max_value):
            if self._history.is_minimalizing():
                self._best_para = min_para
            else:
                self._best_para = max_para

        self._strategy.update(self._individuals)
        self._individuals = self._strategy.generate(creator.Individual)

        for ind in self._individuals:
            ind.fitness.values = (float("nan"),)

        return avg, min_value, max_value, self._history.best

    def get_ind(self, index: int):
        if index >= self._strategy.lambda_:
            raise "'index' >= self._strategy.lambda_"
        ind = self._individuals[index]
        return ind

    def get_best_para(self) -> numpy.ndarray:
        return copy.deepcopy(self._best_para)

    def get_best_score(self) -> float:
        return self._history.best

    def get_history(self) -> Hist:
        return copy.deepcopy(self._history)

    def get_lambda(self):
        return self._strategy.lambda_


class CMAES:
    def __init__(self, env: EnvInterface, generation, population, sigma=0.3, minimalize=True):
        self._base = _BaseCMAES(env.dim(), population, sigma, minimalize)
        self._env = env
        self._generation = generation
        self._start_handler = default_start_handler
        self._end_handler = default_end_handler

    def get_best_para(self) -> numpy.ndarray:
        return self._base.get_best_para()

    def get_best_score(self) -> float:
        return self.get_best_score()

    def get_history(self) -> Hist:
        return self._base.get_history()

    def set_start_handler(self, handler=default_start_handler):
        self._start_handler = handler

    def set_end_handler(self, handler=default_end_handler):
        self._end_handler = handler

    def optimize(self):
        start_time = datetime.datetime.now()
        for gen in range(1, self._generation + 1):
            self._start_handler(gen, self._generation, start_time)

            for i in range(0, self._base.get_lambda()):
                ind = self._base.get_ind(i)
                score = self._env.calc(ind)
                ind.fitness.values = (score,)

            avg, min_value, max_value, best = self._base.generate()

            finish_time = datetime.datetime.now()
            self._end_handler(
                self._base.get_lambda(), gen, self._generation,
                start_time, finish_time,
                avg, min_value, max_value, best
            )
            start_time = copy.copy(finish_time)


class ServerCMAES:
    def __init__(self, env: EnvInterface, generation, population, sigma=0.3, minimalize=True):
        self._base = _BaseCMAES(env.dim(), population, sigma, minimalize)
        self._env = env
        self._generation = generation
        self._start_handler = default_start_handler
        self._end_handler = default_end_handler

    def get_best_para(self) -> numpy.ndarray:
        return self._base.get_best_para()

    def get_best_score(self) -> float:
        return self.get_best_score()

    def get_history(self) -> Hist:
        return self._base.get_history()

    def set_start_handler(self, handler=default_start_handler):
        self._start_handler = handler

    def set_end_handler(self, handler=default_end_handler):
        self._end_handler = handler

    def optimize(self, port):
        import socket
        import threading

        soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        soc.bind(("localhost", port))
        soc.listen(2)

        start_time = datetime.datetime.now()
        for gen in range(1, self._generation + 1):
            while True:
                self._start_handler(gen, self._generation, start_time)

                thread_handles = []
                for i in range(0, self._base.get_lambda()):
                    ind = self._base.get_ind(i)
                    if not math.isnan(ind.fitness.values[0]):
                        print("CALCULATED", i, ind.fitness.values[0])
                        continue

                    conn, _addr = soc.accept()

                    def server_proc(individual, connection):
                        import struct
                        buf = [self._env.save()]
                        buf.extend([struct.pack("<d", x) for x in individual])
                        connection.send(b''.join(buf))
                        received = connection.recv(1024)
                        score = struct.unpack("<d", received)[0]
                        individual.fitness.values = (score,)

                    handle = threading.Thread(target=lambda: server_proc(ind, conn))
                    handle.start()
                    thread_handles.append(handle)

                for th in thread_handles:
                    th.join()

                gen_result = self._base.generate()
                if gen_result is False:
                    print("ERROR HAPPEN AT A CLIENT.")
                    continue
                avg, min_value, max_value, best = gen_result

                finish_time = datetime.datetime.now()
                self._end_handler(
                    self._base.get_lambda(), gen, self._generation,
                    start_time, finish_time,
                    avg, min_value, max_value, best
                )
                start_time = copy.copy(finish_time)

                break


class ClientCMAES:
    def __init__(self, address, port, env: EnvInterface, buf_size: int = 1024):
        self._address = address
        self._port = port
        self._buf_size = buf_size
        self._env = env

    def optimize(self):
        import socket
        import struct

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((self._address, self._port))

            received = sock.recv(self._buf_size)
            print(f"receive data size : {len(received)}/{self._buf_size}")

            env_size = self._env.load(received)
            para = [struct.unpack("<d", received[i:i + 8])[0] for i in range(env_size, len(received), 8)]

            score = self._env.calc(para)

            sock.send(struct.pack("<d", score))
            print(f"score : {score}")

            sock.close()

        except Exception as e:
            sock.close()
            return e

        return None
