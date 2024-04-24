from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.sequence = nn.Sequential(
            nn.Conv1d(3, 20, 8),  # out[20,58]
            nn.MaxPool1d(4),  # out[20,14]
            nn.Tanh(),
            nn.Conv1d(20, 20, 4),  # out[20,11]
            nn.MaxPool1d(2),  # out[20,5]
            nn.Tanh(),
            nn.Flatten(0, -1),  # out[100]
            nn.Linear(100, 50),
            nn.Tanh(),
            nn.Linear(50, 25),
            nn.Tanh(),
            nn.Linear(25, 10),
            nn.Tanh(),
            nn.Linear(10, 5),
            nn.Tanh(),
            nn.Linear(5, 4),
            nn.Tanh(),
            nn.Linear(4, 3),
            nn.Tanh(),
            nn.Linear(3, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        for layer in self.sequence:
            x = layer.forward(x)
        return x
