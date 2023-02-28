from collections.abc import Sequence
from studyLib.nn_tools import interface, la


class ClipLayer(interface.CalcActivator):
    def __init__(self, num_node: int, min_value: float = None, max_value: float = None):
        super().__init__(num_node)
        self._min_value = min_value
        self._max_value = max_value

    def calc(self, input_: la.ndarray, output: la.ndarray) -> int:
        la.clip(input_, self._min_value, self._max_value, out=output)
        return self.num_node

    def num_dim(self) -> int:
        return 0

    def load(self, offset: int, array: Sequence) -> int:
        return 0

    def save(self, array: list) -> None:
        pass
