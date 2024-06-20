import torch
from torch import nn
import numpy as np


class Debugger:
    class Investigator(nn.Module):
        def __init__(self, debug):
            super(Debugger.Investigator, self).__init__()
            self.requires_grad_(False)
            self.buf: np.ndarray | None = None if not debug else np.zeros(1)

        def forward(self, input_: torch.Tensor):
            if self.buf is not None:
                i = input_.detach().numpy()
                if self.buf.shape != i.shape:
                    self.buf = np.zeros(i.shape)
                np.copyto(self.buf, i)
            return input_

    def __init__(self, debug=False):
        self._debug = debug
        self._investigator: dict[str, Debugger.Investigator] = {}

    def create_investigator(self, name: str) -> Investigator:
        i = Debugger.Investigator(self._debug)
        if name in self._investigator.keys():
            raise "[Debugger Error] 'name' is used already."
        self._investigator[name] = i
        return i

    def is_debug_mode(self) -> bool:
        return self._debug

    def get_buf(self) -> dict[str, np.ndarray]:
        res = {}
        for key, value in self._investigator.items():
            res[key] = np.copy(value.buf)
        return res
