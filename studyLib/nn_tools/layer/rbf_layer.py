from collections.abc import Sequence
from studyLib.nn_tools import interface, la


class GaussianRadialBasisLayer(interface.CalcLayer):
    def __init__(self, num_node: int):
        super().__init__(num_node)
        self.centroid = la.zeros(num_node)
        self.weights = la.zeros(num_node)

    def init(self, num_input: int) -> None:
        pass

    def calc(self, input_: la.ndarray) -> la.ndarray:
        d = la.linalg.norm(self.weights * (self.centroid - input_), ord=2)
        return

    def num_dim(self) -> int:
        return 2 * self.num_node

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
