import abc
import enum

import torch

from libs.optimizer import Individual

from .robot_input import RobotInput


class BrainJudgement(enum.Enum):
    STOP = 0
    FORWARD = 1
    BACK = 2
    TURN_RIGHT = 3
    TURN_LEFT = 4
    SECRETION = 5


class BrainInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def think(self, input_: RobotInput) -> torch.Tensor:
        raise NotImplementedError


class BrainBuilder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def build(self, para: Individual) -> BrainInterface:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def get_dim() -> int:
        raise NotImplementedError


class DiscreteOutput(BrainInterface):
    @abc.abstractmethod
    def convert(self, output: torch.Tensor) -> BrainJudgement:
        raise NotImplementedError
