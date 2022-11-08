from collections.abc import Sequence
from studyLib.nn_tools import interface, la


class AffineLayer(interface.CalcLayer):
    def __init__(self, num_node: int):
        super().__init__(num_node)
        self.weights = la.zeros((0, 0))
        self.bias = la.zeros(0)
        self._num_input = 0

    def init(self, num_input: int) -> None:
        self._num_input = num_input
        self.weights = la.zeros((self.num_node, num_input))
        self.bias = la.zeros(self.num_node)

    def calc(self, input_: la.ndarray) -> la.ndarray:
        return self.bias + la.dot(self.weights, input_)

    def num_dim(self) -> int:
        return (self._num_input + 1) * self.num_node

    def load(self, offset: int, array: Sequence) -> int:
        s = offset
        e = s + self._num_input * self.num_node
        self.weights = la.array(array[s:e]).reshape((self.num_node, self._num_input))
        s = e
        e = s + self.num_node
        self.bias = la.array(array[s:e])
        return e - offset

    def save(self, array: list) -> None:
        array.extend(self.weights.ravel())
        array.extend(self.bias)
