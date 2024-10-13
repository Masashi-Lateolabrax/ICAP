import abc
import multiprocessing as mp

from .individual import Individual
from .task_interface import TaskGenerator


class ProcInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, gen: int, thread_id: int, individuals: list[Individual], task_generator: TaskGenerator):
        raise NotImplemented()

    @abc.abstractmethod
    def finished(self) -> bool:
        raise NotImplemented()

    @abc.abstractmethod
    def join(self) -> (int, int):
        raise NotImplemented()


class OneThreadProc(ProcInterface):
    def __init__(self, gen: int, thread_id: int, individuals: list[Individual], task_generator: TaskGenerator):
        self.gen = gen
        self.thread_id = thread_id
        for ind in individuals:
            task = task_generator.generate(ind)
            score = task.run()
            ind.fitness.values = (score,)
            ind.dump = task.get_dump_data()

    def finished(self) -> bool:
        return True

    def join(self) -> (int, int):
        return self.gen, self.thread_id


def _multi_proc_func(individuals: list[Individual], task_generator: TaskGenerator, queue: mp.Queue):
    for ind in individuals:
        task = task_generator.generate(ind)
        ind.fitness.values = (task.run(),)
        ind.dump = task.get_dump_data()
        queue.put(ind)


class MultiThreadProc(ProcInterface):
    def __init__(self, gen: int, thread_id: int, individuals: list[Individual], task_generator: TaskGenerator):
        self.gen = gen
        self.thread_id = thread_id
        self.n = len(individuals)
        self.individuals = individuals
        self.queue = mp.Queue(len(individuals))
        self.handle = mp.Process(target=_multi_proc_func, args=(individuals, task_generator, self.queue))
        self.handle.start()

    def finished(self) -> bool:
        return self.queue.qsize() == self.n

    def join(self) -> (int, int):
        for origin in self.individuals:
            result = self.queue.get()
            origin.fitness.values = result.fitness.values
        self.handle.join()
        return self.gen, self.thread_id
