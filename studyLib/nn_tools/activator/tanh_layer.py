from collections.abc import Sequence
from studyLib.nn_tools import interface, la


class TanhLayer(interface.CalcActivator):
    def __init__(self, num_node: int):
        super().__init__(num_node)

    def calc(self, input_: la.ndarray) -> la.ndarray:
        return la.tanh(input_)

    def num_dim(self) -> int:
        return 0

    def load(self, offset: int, array: Sequence) -> int:
        return 0

    def save(self, array: list) -> None:
        pass
