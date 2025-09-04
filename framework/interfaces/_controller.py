import abc
from typing import TypeVar, Self

from flax import nnx

from ..types import RobotInputs, RobotOutputs


class ControllerInterface(nnx.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, parameter):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, x: RobotInputs) -> RobotOutputs:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> Self:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def dim() -> int:
        raise NotImplementedError


ControllerT = TypeVar("ControllerT", bound=ControllerInterface)
