import torch

from framework.prelude import *


class Controller(torch.nn.Module):
    def __init__(self, settings: Settings, parameters: Individual = None):
        super(Controller, self).__init__()

        self.action_patterns = torch.from_numpy(settings.Action.PATTERNS).float()

        nray = settings.Robot.DEPTH_SENSOR_NUM_RAYS
        self.pheromone_idx = 2 + 3 + nray  # direction(2) + velocity(3) + depth(nray)
        # num_actions = self.action_patterns.shape[0]

        # Input: direction(2) + velocity(3) + depth(nray) + pheromone(3) + pheromone_1e5(1)
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(2 + 3 + nray + 4, 10),
            torch.nn.Mish(),
            torch.nn.Linear(10, 5),
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
        # action_indices = torch.argmax(action_probs, dim=1)
        # selected_actions = self.action_patterns[action_indices]  # Shape: (num_robots, 3)
        # return selected_actions

        # Extract pheromone concentration (shape: (num_robots, 1))
        pheromone_conc = input_[:, self.pheromone_idx:self.pheromone_idx+1]

        # Rescale to pheromone/1e-5 and clip to [0, 1]
        pheromone_1e5 = torch.clamp(pheromone_conc * 3.5 / 1e-5, 0.0, 1.0)

        # Append as 41st dimension: (num_robots, 40) + (num_robots, 1) -> (num_robots, 41)
        expanded_input = torch.cat([input_, pheromone_1e5], dim=1)

        x = self.sequential(expanded_input)
        return torch.sigmoid(x)
