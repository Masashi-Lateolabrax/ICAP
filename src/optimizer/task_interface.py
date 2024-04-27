import abc


class TaskInterface(metaclass=abc.ABCMeta):
    """
    タスクのインターフェイス．
    """

    @abc.abstractmethod
    def run(self) -> float:
        raise NotImplementedError()


class TaskGenerator(metaclass=abc.ABCMeta):
    """
    タスクを生成するためのクラスです．スレッドセーフでなければなりません．
    """

    @abc.abstractmethod
    def save(self) -> bytes:
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, data: bytes) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def generate(self, para) -> TaskInterface:
        raise NotImplementedError()
