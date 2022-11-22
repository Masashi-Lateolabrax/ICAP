from collections.abc import Sequence
from studyLib.nn_tools import interface, la


class ReluLayer(interface.CalcActivator):
    def __init__(self, num_node: int, threshold: float):
        super().__init__(num_node)
        self.threshold = threshold

    def calc(self, input_: la.ndarray) -> la.ndarray:
        input_ -= self.threshold
        return la.maximum(input_, 0.0)

    def num_dim(self) -> int:
        return 0

    def load(self, offset: int, array: Sequence) -> int:
        return 0

    def save(self, array: list) -> None:
        pass
