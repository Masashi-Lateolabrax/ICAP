from collections.abc import Sequence
from studyLib.nn_tools import interface, la


class GaussianRadialBasisLayer(interface.CalcLayer):
    def __init__(self, num_node: int):
        super().__init__(num_node)
        self.centroid = la.zeros((0, 0))
        self.weights = la.ones(num_node)
        self._num_input = 0

    def init(self, num_input: int) -> None:
        self.centroid = la.zeros((self.num_node, num_input))
        self._num_input = num_input

    def calc(self, input_: la.ndarray) -> la.ndarray:
        sub = self.centroid - input_
        dist = la.linalg.norm(sub, axis=1, ord=2)
        return la.exp(self.weights * dist)

    def num_dim(self) -> int:
        return self.num_node * self._num_input + self.num_node

    def load(self, offset: int, array: Sequence) -> int:
        s = offset
        e = s + self._num_input * self.num_node
        self.centroid = la.array(array[s:e]).reshape((self.num_node, self._num_input))
        s = e
        e = s + self.num_node
        self.weights = la.abs(la.array(array[s:e])) * -1.0
        return e - offset

    def save(self, array: list) -> None:
        array.extend(self.centroid.ravel())
        array.extend(la.abs(self.weights))
