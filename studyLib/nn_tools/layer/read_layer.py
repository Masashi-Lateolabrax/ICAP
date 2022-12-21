from collections.abc import Sequence
from studyLib.nn_tools import interface, la

from studyLib.nn_tools.activator.buf_layer import BufLayer


class ReadLayer(interface.CalcActivator):
    def __init__(self, buf_layer: BufLayer):
        super().__init__(0)
        self._buf = buf_layer.buf
        self._num_input = 0

    def init(self, num_input: int) -> None:
        self._num_input = num_input
        self.num_node = num_input + len(self._buf)

    def calc(self, input_: la.ndarray, output: la.ndarray) -> int:
        output.fill(0.0)
        la.copyto(output[0:self._num_input], input_)
        la.copyto(output[self._num_input:self.num_node], self._buf)
        return self.num_node

    def num_dim(self) -> int:
        return 0

    def load(self, offset: int, array: Sequence) -> int:
        return 0

    def save(self, array: list) -> None:
        pass
