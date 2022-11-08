from collections.abc import Sequence
from studyLib.nn_tools import interface, la


class GaussianLayer(interface.CalcLayer):
    def __init__(self, num_node: int, mu: float, sigma: float):
        super().__init__(num_node)
        self.mu = la.array([mu for _i in range(0, num_node)])
        self.inv_sigma2 = la.array([1.0 / (sigma * sigma) for _i in range(0, num_node)])

    def init(self, num_input: int) -> None:
        pass

    def calc(self, input_: la.ndarray) -> la.ndarray:
        return la.exp(-1.0 * la.power(input_ - self.mu, 2) * self.inv_sigma2)

    def num_dim(self) -> int:
        return 2 * self.num_node

    def load(self, offset: int, array: Sequence) -> int:
        s = offset
        e = s + self.num_node
        self.mu = la.array(array[s:e])
        s = e
        e = s + self.num_node
        self.inv_sigma2 = la.abs(la.array(array[s:e]))
        return e - offset

    def save(self, array: list) -> None:
        array.extend(self.mu)
        array.extend(self.inv_sigma2)
