import abc
from collections.abc import Sequence
from studyLib.nn_tools import la


class CalcLayer(metaclass=abc.ABCMeta):
    def __init__(self, num_node: int):
        self.num_node = num_node

    @abc.abstractmethod
    def init(self, num_input: int) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def calc(self, input_: la.ndarray) -> la.ndarray:
        raise NotImplementedError()

    @abc.abstractmethod
    def num_dim(self) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, offset: int, array: Sequence) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def save(self, array: list) -> None:
        raise NotImplementedError()