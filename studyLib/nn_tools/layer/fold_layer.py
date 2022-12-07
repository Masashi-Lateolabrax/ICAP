from collections.abc import Sequence
from studyLib.nn_tools import interface, la


class AddFoldLayer(interface.CalcLayer):
    def __init__(self, num_node: int):
        super().__init__(num_node)
        self._stride = num_node

    def init(self, num_input: int) -> None:
        pass

    def calc(self, input_: la.ndarray, output: la.ndarray) -> int:
        output.fill(0.0)
        s = 0
        e = la.minimum(s + self._stride, len(input_))
        while e <= len(input_):
            output += input_[s:e]
            s = e
            e = la.minimum(s + self._stride, len(input_))
        return self.num_node

    def num_dim(self) -> int:
        return 0

    def load(self, offset: int, array: Sequence) -> int:
        return 0

    def save(self, array: list) -> None:
        pass
