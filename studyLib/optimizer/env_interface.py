import abc
from studyLib import wrap_mjc, miscellaneous


class EnvInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calc(self, para) -> float:
        raise NotImplementedError()


class MuJoCoEnvInterface(EnvInterface, abc.ABC):
    @abc.abstractmethod
    def calc_and_show(self, para, window: miscellaneous.Window, camera: wrap_mjc.Camera) -> float:
        raise NotImplementedError()


class EnvCreator(metaclass=abc.ABCMeta):
    """
    環境を生成するためのクラスです．スレッドセーフでなければなりません．
    """

    @abc.abstractmethod
    def dim(self) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def save(self) -> bytes:
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, env_data: bytes) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def create(self) -> EnvInterface:
        raise NotImplementedError()


class MuJoCoEnvCreator(EnvCreator, abc.ABC):
    """
    MuJoCoを用いた環境を生成するクラスです．スレッドセーフでなければいけません．
    """

    @abc.abstractmethod
    def create_mujoco_env(self) -> MuJoCoEnvInterface:
        raise NotImplementedError()
