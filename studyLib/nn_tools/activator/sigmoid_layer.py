from collections.abc import Sequence
from studyLib.nn_tools import interface, la


class SigmoidLayer(interface.CalcLayer):
    def __init__(self, num_node: int, a: float = 1.0):
        super().__init__(num_node)
        self.alpha = la.abs(a)

    def init(self, num_input: int) -> None:
        pass

    def calc(self, input_: la.ndarray) -> la.ndarray:
        return 1.0 / (1.0 + la.exp(-self.alpha * input_))

    def num_dim(self) -> int:
        return 0

    def load(self, offset: int, array: Sequence) -> int:
        return 0

    def save(self, array: list) -> None:
        pass
