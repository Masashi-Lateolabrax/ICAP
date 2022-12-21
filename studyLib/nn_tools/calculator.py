from studyLib.nn_tools import interface, la


class Calculator:
    class CalcBuffer:
        def __init__(self):
            self.buf1: la.ndarray = la.zeros(0)
            self.buf2: la.ndarray = la.zeros(0)

    def __init__(self, num_input: int, buf: CalcBuffer = None):
        self._layer: list[interface.CalcLayer] = []
        self._num_input = num_input
        self._buf = Calculator.CalcBuffer() if buf is None else buf

    def get_layer(self, i: int) -> interface.CalcLayer:
        return self._layer[i]

    def add_layer(self, layer: interface.CalcLayer):
        ni = self._layer[-1].num_node if len(self._layer) > 0 else self._num_input
        self._layer.append(layer)
        self._layer[-1].init(ni)

    def num_dim(self) -> int:
        n = 0
        for layer in self._layer:
            n += layer.num_dim()
        return n

    def num_input(self) -> int:
        return self._num_input

    def num_output(self) -> int:
        return self._layer[-1].num_node

    def calc(self, input_):
        size = len(input_)
        if len(self._buf.buf1) < size:
            self._buf.buf1 = la.zeros(size)
        la.copyto(self._buf.buf1[0:size], input_)

        for i, layer in enumerate(self._layer):
            if i % 2 == 0:
                if len(self._buf.buf2) < layer.num_node:
                    self._buf.buf2 = la.zeros(layer.num_node)
                size = layer.calc(self._buf.buf1[0:size], self._buf.buf2[0:layer.num_node])
            else:
                if len(self._buf.buf1) < layer.num_node:
                    self._buf.buf1 = la.zeros(layer.num_node)
                size = layer.calc(self._buf.buf2[0:size], self._buf.buf1[0:layer.num_node])

        if len(self._layer) % 2 == 0:
            return self._buf.buf1[0:size]
        else:
            return self._buf.buf2[0:size]

    def load(self, array):
        dim = self.num_dim()
        if len(array) != dim:
            print(f"[WARNING] The Passed Calculator Parameter Dimension is {len(array)} "
                  f"but The Calculater Dimension is {dim}.")
        offset = 0
        for layer in self._layer:
            offset += layer.load(offset, array)

    def save(self, array):
        for layer in self._layer:
            layer.save(array)
