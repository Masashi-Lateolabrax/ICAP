from collections.abc import Sequence
from studyLib.nn_tools import interface, la, Calculator


class ParallelLayer(interface.CalcLayer):
    def __init__(self, calcs: list[Calculator]):
        num_node = 0
        for c in calcs:
            num_node += c.num_output()
        super().__init__(num_node)
        self.calcs = calcs
        self._input_buf = la.zeros(0)

    def init(self, num_input: int) -> None:
        self._input_buf = la.zeros(num_input)

    def calc(self, input_: la.ndarray, output: la.ndarray) -> int:
        s = 0
        for c in self.calcs:
            e = s + c.num_output()
            la.copyto(self._input_buf, input_)
            la.copyto(output[s:e], c.calc(self._input_buf))
            s = e
        return self.num_node

    def num_dim(self) -> int:
        d = 0
        for c in self.calcs:
            d += c.num_dim()
        return d

    def load(self, offset: int, array: Sequence) -> int:
        s = offset
        e = 0
        for c in self.calcs:
            e = s + c.num_dim()
            sub_array = array[s:e]
            c.load(sub_array)
            s = e
        return e - offset

    def save(self, array: list) -> None:
        for c in self.calcs:
            c.save(array)
