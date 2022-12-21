from collections.abc import Sequence
from studyLib.nn_tools import interface, la, Calculator


class BlockLayer(interface.CalcLayer):

    def __init__(self):
        super().__init__(0)

    def init(self, num_input: int) -> None:
        pass

    def calc(self, input_: la.ndarray, output: la.ndarray) -> int:
        return self.num_node

    def num_dim(self) -> int:
        return 0

    def load(self, offset: int, array: Sequence) -> int:
        return 0

    def save(self, array: list) -> None:
        pass
