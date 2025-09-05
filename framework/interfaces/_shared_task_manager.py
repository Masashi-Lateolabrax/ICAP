import abc

from ..prelude import *


class SharedTaskManagerTrait(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def sync_(self):
        raise NotImplementedError

    @abc.abstractmethod
    def check_heartbeat(self) -> list[int]:  # returns list of alive client IDs
        raise NotImplementedError

    @abc.abstractmethod
    def close_dead_clients(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def get_client_statistics(self) -> dict[int, ClientStatistics]:
        raise NotImplementedError
