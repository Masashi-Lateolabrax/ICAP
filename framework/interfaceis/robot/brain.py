import abc
import enum


class BrainJudgement(enum.Enum):
    STOP = 0
    FORWARD = 1
    BACK = 2
    TURN_RIGHT = 3
    TURN_LEFT = 4
    SECRETION = 5


class BrainInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, para):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def get_dim() -> int:
        raise NotImplementedError

    def think(self, input_) -> BrainJudgement:
        raise NotImplementedError
