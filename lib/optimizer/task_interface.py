import abc
import pickle

import mujoco
from mujoco_xml_generator.utils import DummyGeom


class TaskInterface(metaclass=abc.ABCMeta):
    """
    タスクのインターフェイス．
    """

    @abc.abstractmethod
    def run(self) -> float:
        raise NotImplementedError()


class MjcTaskInterface(TaskInterface):
    """
    mujocoを用いるタスクのインターフェイス．
    """

    @abc.abstractmethod
    def get_model(self) -> mujoco.MjModel:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_data(self) -> mujoco.MjData:
        raise NotImplementedError()

    @abc.abstractmethod
    def calc_step(self) -> float:
        raise NotImplementedError()

    def get_dummies(self) -> list[DummyGeom]:
        return []


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
    def generate(self, para, debug=False) -> TaskInterface:
        raise NotImplementedError()
