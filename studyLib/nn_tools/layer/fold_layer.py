from collections.abc import Sequence
from studyLib.nn_tools import interface, la


class FoldLayer(interface.CalcLayer):
    def __init__(self, func, num_node: int, stride: int = None):
        super().__init__(num_node)
        self._stride = num_node if stride is None else stride
        self._func = func

    def init(self, num_input: int) -> None:
        pass

    def calc(self, input_: la.ndarray, output: la.ndarray) -> int:
        if len(input_) <= len(output):
            output.fill(0.0)
            la.copyto(output[0:len(input_)], input_)
            return self.num_node

        la.copyto(output, input_[0:self.num_node])
        offset = self._stride
        while offset + self.num_node <= len(input_):
            self._func(input_[offset:(offset + self.num_node)], output)
            offset += self._stride
        return self.num_node

    def num_dim(self) -> int:
        return 0

    def load(self, offset: int, array: Sequence) -> int:
        return 0

    def save(self, array: list) -> None:
        pass


class AddFoldLayer(FoldLayer):
    @staticmethod
    def _add_func(values: la.ndarray, output: la.ndarray):
        output += values

    def __init__(self, num_node: int, stride: int = None):
        super().__init__(AddFoldLayer._add_func, num_node, stride)


class MulFoldLayer(FoldLayer):
    @staticmethod
    def _mul_func(values: la.ndarray, output: la.ndarray):
        output *= values

    def __init__(self, num_node: int, stride: int = None):
        super().__init__(MulFoldLayer._mul_func, num_node, stride)
