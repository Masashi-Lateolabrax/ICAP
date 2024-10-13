import abc

from deap.cma import Strategy

from ..individual import Individual


class Logger(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def log(
            self,
            num_error, avg, min_score, min_para, max_score, max_para, best_para,
            individuals: list[Individual],
            strategy: Strategy
    ):
        raise NotImplemented

    @abc.abstractmethod
    def save_tmp(self, gen):
        raise NotImplemented
