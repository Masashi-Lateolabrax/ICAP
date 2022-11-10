import abc
import datetime
import numpy
import copy

from deap import cma, creator, base

from studyLib.wrap_mjc import Camera
from studyLib.miscellaneous import Window, Recorder
from studyLib.optimizer.history import Hist
from studyLib.optimizer.env_interface import EnvInterface, MuJoCoEnvInterface


def default_start_handler(gen, generation, start_time):
    print(f"[{start_time}] start {gen} gen. ({gen}/{generation}={float(gen) / generation * 100.0}%)")


def default_end_handler(population, gen, generation, start_time, fin_time, avg, min_v, max_v, best):
    elapse = float((fin_time - start_time).total_seconds())
    spd = population / elapse
    e = datetime.timedelta(seconds=(generation - gen) * elapse)
    print(
        f"[{fin_time}] finish {gen} gen. speed[ind/s]:{spd}, avg:{avg}, min:{min_v}, max:{max_v}, best:{best}, etr:{e}"
    )


class _CalcHandler(metaclass=abc.ABCMeta):
    def __init__(self, index: int):
        self.index = index

    def get_index(self) -> int:
        return self.index

    @abc.abstractmethod
    def join(self) -> float:
        raise NotImplementedError()


class _Calculator(metaclass=abc.ABCMeta):
    def create(self, index, ind) -> _CalcHandler:
        raise NotImplementedError()


class _BaseCMAES:
    def __init__(self, dim: int, population: int, sigma=0.3, minimalize=True):
        self._best_para = numpy.zeros(0)
        self._history = Hist(minimalize)
        self._start_handler = default_start_handler
        self._end_handler = default_end_handler

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

    def _generate_new_generation(self) -> (float, float, float, numpy.ndarray, float):
        avg = 0.0
        min_value = float("inf")
        max_value = -float("inf")
        good_para = numpy.zeros(0)

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
            self._best_para = good_para.copy()

        self._strategy.update(self._individuals)
        self._individuals = self._strategy.generate(creator.Individual)

        for ind in self._individuals:
            ind.fitness.values = (float("nan"),)

        return avg, min_value, max_value, good_para, self._history.best

    def optimize_current_generation(self, gen: int, generation: int, calculator: _Calculator) -> numpy.ndarray:
        start_time = datetime.datetime.now()
        self._start_handler(gen, generation, start_time)

        res = None
        while res is None:
            handles = []
            for i, ind in enumerate(self._individuals):
                if not numpy.isnan(ind.fitness.values[0]):
                    print("CALCULATED", i, ind.fitness.values[0])
                    continue
                handles.append(calculator.create(i, ind))

            for h in handles:
                score = h.join()
                self._individuals[h.get_index()].fitness.values = (score,)

            res = self._generate_new_generation()

        avg, min_value, max_value, good_para, best = res

        finish_time = datetime.datetime.now()
        self._end_handler(
            self.get_lambda(), gen, generation,
            start_time, finish_time,
            avg, min_value, max_value, best
        )

        return good_para

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

    def set_start_handler(self, handler=default_start_handler):
        self._start_handler = handler

    def set_end_handler(self, handler=default_end_handler):
        self._end_handler = handler


class CMAES:
    def __init__(self, dim: int, generation, population, sigma=0.3, minimalize=True):
        self._base = _BaseCMAES(dim, population, sigma, minimalize)
        self._generation = generation

    def get_best_para(self) -> numpy.ndarray:
        return self._base.get_best_para()

    def get_best_score(self) -> float:
        return self.get_best_score()

    def get_history(self) -> Hist:
        return self._base.get_history()

    def set_start_handler(self, handler=default_start_handler):
        self._base.set_start_handler(handler)

    def set_end_handler(self, handler=default_end_handler):
        self._base.set_end_handler(handler)

    def optimize(self, env: EnvInterface):
        class CH(_CalcHandler):
            def __init__(self, index: int, score: float):
                super().__init__(index)
                self.score = score

            def join(self) -> float:
                return self.score

        class C(_Calculator):
            def create(self, index, ind) -> _CalcHandler:
                return CH(index, env.calc(ind))

        for gen in range(1, self._generation + 1):
            self._base.optimize_current_generation(gen, self._generation, C())

    def optimize_with_recoding_min(self, env: MuJoCoEnvInterface, window: Window, camera: Camera):
        class CH(_CalcHandler):
            def __init__(self, index: int, score: float):
                super().__init__(index)
                self.score = score

            def join(self) -> float:
                return self.score

        class C(_Calculator):
            def create(self, index, ind) -> _CalcHandler:
                return CH(index, env.calc(ind))

        for gen in range(1, self._generation + 1):
            good_para = self._base.optimize_current_generation(gen, self._generation, C())

            time = datetime.datetime.now()
            recorder = Recorder(f"{time.strftime('%y%m%d_%H%M%S')}.mp4", 30, 640, 480)
            window.set_recorder(recorder)
            env.calc_and_show(good_para, window, camera)
            window.set_recorder(None)


def _server_proc(env, individual, connection):
    import struct
    buf = [env.save()]
    buf.extend([struct.pack("<d", x) for x in individual])
    connection.send(b''.join(buf))
    received = connection.recv(1024)
    score = struct.unpack("<d", received)[0]
    individual.fitness.values = (score,)


class ServerCMAES:
    def __init__(self, dim: int, generation, population, sigma=0.3, minimalize=True):
        self._base = _BaseCMAES(dim, population, sigma, minimalize)
        self._generation = generation

    def get_best_para(self) -> numpy.ndarray:
        return self._base.get_best_para()

    def get_best_score(self) -> float:
        return self.get_best_score()

    def get_history(self) -> Hist:
        return self._base.get_history()

    def set_start_handler(self, handler=default_start_handler):
        self._base.set_start_handler(handler)

    def set_end_handler(self, handler=default_end_handler):
        self._base.set_end_handler(handler)

    def optimize(self, port: int, env: EnvInterface):
        import socket
        import threading

        soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        soc.bind(("0.0.0.0", port))
        soc.listen(2)

        class CH(_CalcHandler):
            def __init__(self, index: int, ind, handle):
                super().__init__(index)
                self.ind = ind
                self.handle = handle

            def join(self) -> float:
                self.handle.join()
                return self.ind.fitness.values[0]

        class C(_Calculator):
            def create(self, index, ind) -> _CalcHandler:
                conn, _addr = soc.accept()
                handle = threading.Thread(target=lambda: _server_proc(env, ind, conn))
                handle.start()
                return CH(index, ind, handle)

        for gen in range(1, self._generation + 1):
            self._base.optimize_current_generation(gen, self._generation, C())

    def optimize_with_recoding_min(self, port: int, env: MuJoCoEnvInterface, window: Window, camera: Camera):
        import socket
        import threading

        soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        soc.bind(("0.0.0.0", port))
        soc.listen(2)

        class CH(_CalcHandler):
            def __init__(self, index: int, ind, handle):
                super().__init__(index)
                self.ind = ind
                self.handle = handle

            def join(self) -> float:
                self.handle.join()
                return self.ind.fitness.values[0]

        class C(_Calculator):
            def create(self, index, ind) -> _CalcHandler:
                conn, _addr = soc.accept()
                handle = threading.Thread(target=lambda: _server_proc(env, ind, conn))
                handle.start()
                return CH(index, ind, handle)

        for gen in range(1, self._generation + 1):
            self._base.optimize_current_generation(gen, self._generation, C())


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
