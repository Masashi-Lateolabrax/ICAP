import abc
from studyLib import wrap_mjc, miscellaneous


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


class MuJoCoEnvInterface(EnvInterface, abc.ABC):
    @abc.abstractmethod
    def calc_and_show(self, para, window: miscellaneous.Window, camera: wrap_mjc.Camera) -> float:
        raise NotImplementedError()
