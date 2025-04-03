import abc
import enum

from libs.optimizer import Individual


class BrainJudgement(enum.Enum):
    STOP = 0
    FORWARD = 1
    BACK = 2
    TURN_RIGHT = 3
    TURN_LEFT = 4
    SECRETION = 5


class BrainInterface(metaclass=abc.ABCMeta):
    def think(self, input_) -> BrainJudgement:
        raise NotImplementedError


class BrainBuilder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def build(self, para: Individual) -> BrainInterface:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def get_dim() -> int:
        raise NotImplementedError
