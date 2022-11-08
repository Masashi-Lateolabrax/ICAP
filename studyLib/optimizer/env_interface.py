import abc
from studyLib.miscellaneous import Window


class EnvInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def dim(self) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def calc(self, para) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def save(self) -> bytes:
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, env_data: bytes) -> int:
        raise NotImplementedError()
