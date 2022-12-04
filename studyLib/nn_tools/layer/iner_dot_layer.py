from collections.abc import Sequence
from studyLib.nn_tools import interface, la


class InnerDotLayer(interface.CalcLayer):
    def __init__(self, num_node: int):
        super().__init__(num_node)
        self.weights = la.zeros((0, 0))
        self._num_input = 0

    def init(self, num_input: int) -> None:
        self.weights = la.zeros((self.num_node, num_input))
        self._num_input = num_input

    def calc(self, input_: la.ndarray, output: la.ndarray) -> int:
        la.dot(self.weights, input_, output)
        return self.num_node

    def num_dim(self) -> int:
        return self.num_node * self._num_input

    def load(self, offset: int, array: Sequence) -> int:
        s = offset
        e = s + self._num_input * self.num_node
        self.weights = la.array(array[s:e]).reshape((self.num_node, self._num_input))
        return e - offset

    def save(self, array: list) -> None:
        array.extend(self.weights.ravel())
