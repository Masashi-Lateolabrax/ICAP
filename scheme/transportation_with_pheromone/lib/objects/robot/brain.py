import abc
import enum

import torch


class BrainInputInterface(metaclass=abc.ABCMeta):
    def get(self) -> torch.Tensor:
        raise NotImplementedError


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

    def think(self, input_: BrainInputInterface) -> BrainJudgement:
        raise NotImplementedError
