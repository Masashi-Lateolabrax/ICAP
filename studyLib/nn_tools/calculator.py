from studyLib.nn_tools import interface


class Calculator:
    def __init__(self, num_input: int):
        self._layer: list[interface.CalcLayer] = []
        self.num_input = num_input

    def add_layer(self, layer: interface.CalcLayer):
        ni = self.num_input
        if len(self._layer) > 0:
            ni = self._layer[-1].num_node

        self._layer.append(layer)
        self._layer[-1].init(ni)

    def num_dim(self):
        n = 0
        for layer in self._layer:
            n += layer.num_dim()
        return n

    def calc(self, input_):
        buf1 = input_.copy()
        buf2 = None
        for i, layer in enumerate(self._layer):
            if i % 2 == 0:
                buf2 = layer.calc(buf1)
            else:
                buf1 = layer.calc(buf2)
        if len(self._layer) % 2 == 0:
            return buf1
        else:
            return buf2

    def load(self, array):
        offset = 0
        for layer in self._layer:
            offset += layer.load(offset, array)

    def save(self, array):
        for layer in self._layer:
            layer.save(array)
