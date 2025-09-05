import abc

from ..prelude import *


class ClientTrait(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def request_task(self, n: int) -> list[Task]:
        raise NotImplementedError

    @abc.abstractmethod
    def return_results(self, tasks: list[Task]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def send_statistics(self, statistics: ClientStatistics) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def send_error(self, error: str) -> None:
        raise NotImplementedError
