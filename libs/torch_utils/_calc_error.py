import torch
import torch.nn as nn


class PseudoCalcError(nn.Module):
    def __init__(self):
        super(PseudoCalcError, self).__init__()

        self.register_buffer('epsilon_float32', torch.tensor(1e-7, dtype=torch.float32))
        self.register_buffer('epsilon_float64', torch.tensor(1e-15, dtype=torch.float64))

    def forward(self, x):
        if x.dtype == torch.float32:
            epsilon = self.epsilon_float32
        elif x.dtype == torch.float64:
            epsilon = self.epsilon_float64
        else:
            raise ValueError("Unsupported dtype")
        epsilon = epsilon.to(x.device)

        epsilon = epsilon * (10 ** torch.floor(torch.log10(torch.abs(x))))
        noise = epsilon * torch.randint_like(x, -1, 2)
        return x + noise
