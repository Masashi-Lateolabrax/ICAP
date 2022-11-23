from collections.abc import Sequence

from studyLib.nn_tools import interface, la


class IsMaxLayer(interface.CalcActivator):
    def __init__(self, num_node: int):
        super().__init__(num_node)

    def calc(self, input_: la.ndarray) -> la.ndarray:
        max_value = la.max(input_)
        mask = input_ >= max_value
        input_[la.logical_not(mask)] = 0.0
        input_[mask] = 1.0 / la.count_nonzero(mask)
        return input_

    def num_dim(self) -> int:
        return 0

    def load(self, offset: int, array: Sequence) -> int:
        return 0

    def save(self, array: list) -> None:
        pass


if __name__ == '__main__':
    def test():
        is_min_layer = IsMaxLayer(5)
        sample = la.array([1.0, 3.0, 6.0, 1.9, 7.0])
        print(is_min_layer.calc(sample))


    test()