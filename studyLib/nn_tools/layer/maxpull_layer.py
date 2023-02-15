from collections.abc import Sequence
from studyLib.nn_tools import interface, la


class MaxPullLayer(interface.CalcLayer):
    def __init__(self):
        super().__init__(1)
        self._num_input = 0

    def init(self, num_input: int) -> None:
        self._num_input = num_input

    def calc(self, input_: la.ndarray, output: la.ndarray) -> int:
        output[0] = la.max(input_)
        return self.num_node

    def num_dim(self) -> int:
        return 0

    def load(self, offset: int, array: Sequence) -> int:
        return 0

    def save(self, array: list) -> None:
        pass
