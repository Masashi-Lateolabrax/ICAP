from collections.abc import Sequence
from studyLib.nn_tools import interface, la


class SoftmaxLayer(interface.CalcActivator):
    def __init__(self, num_node: int):
        super().__init__(num_node)

    def calc(self, input_: la.ndarray, output: la.ndarray) -> int:
        if len(output) < self.num_node:
            output.resize((self.num_node,))
        output = output[0:self.num_node]

        la.exp(input_, out=output)
        s = la.sum(output)
        output /= s

        return self.num_node

    def num_dim(self) -> int:
        return 0

    def load(self, offset: int, array: Sequence) -> int:
        return 0

    def save(self, array: list) -> None:
        pass
