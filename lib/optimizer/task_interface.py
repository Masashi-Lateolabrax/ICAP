import abc
import pickle


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

    def save(self) -> bytes:
        return pickle.dumps(self)

    def load(self, byte_data: bytes) -> int:
        new_instance = pickle.loads(byte_data)
        self.__dict__.update(new_instance.__dict__)
        return len(byte_data)

    @abc.abstractmethod
    def generate(self, para) -> TaskInterface:
        raise NotImplementedError()
