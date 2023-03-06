from collections.abc import Sequence
from studyLib.nn_tools import interface, la, Calculator


class ParallelLayer(interface.CalcLayer):
    def __init__(self, layers: list[list[interface.CalcLayer]], buf: Calculator.CalcBuffer = None):
        self._buf = Calculator.CalcBuffer() if buf is None else buf
        self._layers = layers
        self.calcs: list[Calculator] = []
        self._input_buf = la.zeros(0)

        num_node = 0
        for layer_for_calc in layers:
            for layer in layer_for_calc:
                if layer is Calculator:
                    raise "You passed a Calculator to a ParallelLayer. It is not supported!"
            num_node += layer_for_calc[-1].num_node
        super().__init__(num_node)

    def init(self, num_input: int) -> None:
        for layer_for_calc in self._layers:
            calc = Calculator(num_input, self._buf)
            for layer in layer_for_calc:
                calc.add_layer(layer)
            self.calcs.append(calc)
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
