import torch

from framework.prelude import *


class Controller(torch.nn.Module):
    def __init__(self, settings: Settings, parameters: Individual = None):
        super(Controller, self).__init__()

        self.action_patterns = torch.from_numpy(settings.Action.PATTERNS).float()

        nray = settings.Robot.DEPTH_SENSOR_NUM_RAYS
        # num_actions = self.action_patterns.shape[0]
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(2 + 3 + nray, 5),
            torch.nn.Mish(),
            torch.nn.Linear(5, 5),
            torch.nn.Mish(),
            torch.nn.Linear(5, 5),
            torch.nn.Mish(),
            torch.nn.Linear(5, 3),  # Output action logits
            # torch.nn.Softmax(dim=-1),  # Convert to probability distribution
        )

        if parameters is not None:
            assert len(parameters) == self.dim, "Parameter length does not match the network's parameter count."
            torch.nn.utils.vector_to_parameters(
                torch.tensor(parameters, dtype=torch.float32),
                self.parameters()
            )

    @property
    def dim(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_):
        # action_probs = self.sequential(input_)  # Shape: (num_robots, num_actions)
        # action_indices = torch.multinomial(action_probs, num_samples=1).squeeze(-1)
        # selected_actions = self.action_patterns[action_indices]  # Shape: (num_robots, 3)
        # return selected_actions

        x = self.sequential(input_)
        return torch.sigmoid(x)
