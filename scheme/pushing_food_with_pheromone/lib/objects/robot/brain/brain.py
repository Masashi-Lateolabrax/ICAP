import abc

import enum

from ..input import RobotInput


class BrainJudgement(enum.Enum):
    STOP = 0
    FORWARD = 1
    BACK = 2
    TURN_RIGHT = 3
    TURN_LEFT = 4
    SECRETION = 5


class BrainInterface(metaclass=abc.ABCMeta):
    @staticmethod
    @abc.abstractmethod
    def get_dim() -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def think(self, input_: RobotInput) -> BrainJudgement:
        raise NotImplementedError
