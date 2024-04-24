import abc


class EnvInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calc_step(self) -> float:
        raise NotImplementedError()

    @abc.abstractmethod
    def calc(self) -> float:
        raise NotImplementedError()


class EnvCreator(metaclass=abc.ABCMeta):
    """
    環境を生成するためのクラスです．スレッドセーフでなければなりません．
    """

    @abc.abstractmethod
    def save(self) -> bytes:
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, env_data: bytes) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def create(self, para) -> EnvInterface:
        raise NotImplementedError()
