import torch
import torch.nn as nn
import torch.nn.functional as F

import framework.simulator.objects


class SubNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SubNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation1 = nn.Tanhshrink()

        self.layer2 = nn.Linear(hidden_size, output_size)
        self.activation2 = nn.Tanh()

    def forward(self, x) -> torch.Tensor:
        x = self.layer1.forward(x)
        x = self.activation1.forward(x)
        x = self.layer2.forward(x)
        x = self.activation2.forward(x)
        return x


class GatingNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_subnets):
        super(GatingNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation1 = nn.Tanhshrink()

        self.layer2 = nn.Linear(hidden_size, num_subnets)
        self.activation2 = nn.Softmax(dim=0)

    def forward(self, x) -> torch.Tensor:
        x = self.layer1.forward(x)
        x = self.activation1.forward(x)
        x = self.layer2.forward(x)
        x = self.activation2.forward(x)
        return x


class NeuralNetwork(nn.Module):
    def __init__(self, num_subnets, one_hot=False):
        self.one_hot = one_hot

        super(NeuralNetwork, self).__init__()
        self.subnets = nn.ModuleList([SubNetwork(6, 4, 2) for _ in range(num_subnets)])
        self.gating_network = GatingNetwork(6, 6, num_subnets)

    def forward(self, x: torch.Tensor):
        outputs = torch.stack([subnet(x) for subnet in self.subnets], dim=0)

        gate_weights = self.gating_network.forward(x)
        if self.one_hot:
            gate_weights = F.one_hot(gate_weights.argmax(), num_classes=gate_weights.size(1))

        return gate_weights @ outputs

    def set_para(self, para):
        i = 0
        for p in self.parameters():
            n = p.numel()
            p.data = torch.tensor(
                para[i:i + n], dtype=torch.float32
            ).reshape(p.data.shape)
            i += n

    def num_dim(self):
        res = 0
        for p in self.parameters():
            res += p.numel()
        return res
