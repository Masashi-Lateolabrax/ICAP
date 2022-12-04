from studyLib.nn_tools import interface, la


class Calculator:
    def __init__(self, num_input: int):
        self._layer: list[interface.CalcLayer] = []
        self.num_input = num_input
        self.buf1 = la.zeros(num_input)
        self.buf2 = la.zeros(0)

    def get_layer(self, i: int) -> interface.CalcLayer:
        return self._layer[i]

    def add_layer(self, layer: interface.CalcLayer):
        ni = self._layer[-1].num_node if len(self._layer) > 0 else self.num_input
        self._layer.append(layer)
        self._layer[-1].init(ni)

    def num_dim(self):
        n = 0
        for layer in self._layer:
            n += layer.num_dim()
        return n

    def calc(self, input_):
        size = len(input_)
        la.copyto(self.buf1[0:size], input_)

        for i, layer in enumerate(self._layer):
            if i % 2 == 0:
                if len(self.buf2) < layer.num_node:
                    self.buf2 = la.zeros(layer.num_node)
                size = layer.calc(self.buf1[0:size], self.buf2[0:layer.num_node])
            else:
                if len(self.buf1) < layer.num_node:
                    self.buf1 = la.zeros(layer.num_node)
                size = layer.calc(self.buf2[0:size], self.buf1[0:layer.num_node])

        if len(self._layer) % 2 == 0:
            return self.buf1[0:size]
        else:
            return self.buf2[0:size]

    def load(self, array):
        offset = 0
        for layer in self._layer:
            offset += layer.load(offset, array)

    def save(self, array):
        for layer in self._layer:
            layer.save(array)
