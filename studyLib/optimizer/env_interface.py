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
    def set_window(self, window: miscellaneous.Window) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def set_camera(self, camera: wrap_mjc.Camera) -> None:
        raise NotImplementedError()
