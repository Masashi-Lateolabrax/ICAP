from collections.abc import Sequence
from studyLib.nn_tools import interface, la


class SigmoidLayer(interface.CalcActivator):
    def __init__(self, num_node: int, a: float = 1.0):
        super().__init__(num_node)
        self.alpha = la.abs(a)

    def calc(self, input_: la.ndarray, output: la.ndarray) -> int:
        la.copyto(output, input_)
        buf = input_
        output *= -self.alpha
        la.exp(output, out=buf)
        buf += 1.0
        output.fill(1.0)
        output /= buf

        return self.num_node

    def num_dim(self) -> int:
        return 0

    def load(self, offset: int, array: Sequence) -> int:
        return 0

    def save(self, array: list) -> None:
        pass
