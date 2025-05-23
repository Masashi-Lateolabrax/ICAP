import abc

from deap.cma import Strategy

from ..individual import Individual


class Logger(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def log(
            self,
            num_error, avg, min_idx, max_idx,
            individuals: list[Individual],
            strategy: Strategy
    ):
        raise NotImplemented

    @abc.abstractmethod
    def save_tmp(self, gen):
        raise NotImplemented


class EachGenLogger(Logger, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __getitem__(self, item):
        raise NotImplemented

    @abc.abstractmethod
    def __len__(self):
        raise NotImplemented
