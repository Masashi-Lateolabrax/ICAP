from abc import ABC
from collections.abc import Sequence
from studyLib.nn_tools import interface, la


class _ConvLayer(interface.CalcLayer, ABC):
    def __init__(self, num_window: int, window: interface.CalcLayer, window_size: int, num_pad: int):
        super().__init__(num_window * window.num_node)
        self._window = window
        self._num_window = num_window
        self._window_size = window_size
        self._num_pad = num_pad
        self._num_input = 0

        self._window.init(window_size)

    def num_dim(self) -> int:
        return self._window.num_dim()

    def load(self, offset: int, array: Sequence) -> int:
        return self._window.load(offset, array)

    def save(self, array: list) -> None:
        self._window.save(array)


class Conv1DLayer(_ConvLayer):
    def __init__(
            self,
            num_window: int,
            window_size: int,
            stride: int,
            window: interface.CalcLayer,
            num_pad: int
    ):
        if window_size < num_pad:
            print("'num_pad' is lager than 'window_size'. It is incorrect.")

        super().__init__(num_window, window, window_size, num_pad)
        self.stride = stride
        self._in_buf = la.zeros(self._window_size)

    def init(self, num_input: int) -> None:
        self._num_input = num_input
        stride: float = (self._num_input + self._num_pad * 2 - self._window_size) / (self._num_window - 1)

        if not stride.is_integer() or self.stride != int(stride):
            if not stride.is_integer():
                print(
                    f"Failed to calculate stride size({stride}). "
                    f"({self._num_input}, {self._num_pad}, {self._window_size}, {self._num_window})"
                )
            elif self.stride != int(stride):
                print("Invalid parameters are passed.")

            nw = ((self._num_input + self._num_pad * 2 - self._window_size) / self.stride) + 1
            ws = self._num_input + self._num_pad * 2 - (self._num_window - 1) * self.stride
            np = ((self._num_window - 1) * self.stride + self._window_size - self._num_input) * 0.5
            s = (self._num_input + self._num_pad * 2 - self._window_size) / (self._num_window - 1)

            print("Could it be that ...")
            if nw >= 1 and nw.is_integer():
                print(f"\t'num_window' = {int(nw)}?")
            print(f"\t'window_size' = {ws}?")
            if np >= 0 and np.is_integer():
                print(f"\t'num_pad' = {int(np)}?")
            if s >= 1 and s.is_integer():
                print(f"\t'stride' = {int(s)}?")

            raise "Failed to set Conv1DLayer."

    def calc(self, input_: la.ndarray, output: la.ndarray) -> int:
        input_index = 0
        out_buf_index = 0
        padding_i = int(self._num_pad / self.stride) + 1 if self._num_pad > 0 else 0

        for i in range(self._num_window):
            if i < padding_i:
                np = self._num_pad - i * self.stride
                self._in_buf[:np] = 0
                self._in_buf[np:] = input_[input_index:input_index + self._window_size - np]
                input_index += self.stride - np if self.stride - np > 0 else 0
            elif self._num_window - i <= padding_i:
                np = self._num_pad - (self._num_window - i - 1) * self.stride
                self._in_buf[:-np] = input_[input_index:input_index + self._window_size - np]
                self._in_buf[-np:] = 0
                input_index += self.stride
            else:
                self._in_buf[:] = input_[input_index:input_index + self._window_size]
                input_index += self.stride

            self._window.calc(
                self._in_buf,
                output[out_buf_index:out_buf_index + self._window.num_node]
            )
            out_buf_index += self._window.num_node

        return self._window.num_node * self._num_window
