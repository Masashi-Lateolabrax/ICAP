from collections.abc import Sequence
from studyLib.nn_tools import interface, la


class FilterLayer(interface.CalcLayer):
    def __init__(self, pass_indexes: list[int]):
        super().__init__(len(pass_indexes))
        self._mask = []
        self._pass_indexes = pass_indexes

    def init(self, num_input: int) -> None:
        self._mask = la.array([False] * num_input)
        for i in self._pass_indexes:
            self._mask[i] = True

    def calc(self, input_: la.ndarray, output: la.ndarray) -> int:
        la.copyto(output, input_[self._mask])
        return self.num_node

    def num_dim(self) -> int:
        return 0

    def load(self, offset: int, array: Sequence) -> int:
        return 0

    def save(self, array: list) -> None:
        pass
