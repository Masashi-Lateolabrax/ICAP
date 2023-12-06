import torch


class ReplayMemory:
    class _Episode:
        def __init__(self, state_dim):
            self.is_valid = False
            self.state = torch.Tensor([0.0] * state_dim)
            self.action_id = 0
            self.reward = 0.0
            self.q = 0.0
            self.max_q = 0.0

        def set(self, state, reward: float, max_q: float):
            self.is_valid = True
            self.state[:] = state
            self.reward = reward
            self.max_q = max_q

        def disable(self):
            self.is_valid = False

    def __init__(self, state_dim: int, length: int):
        self.index = 0
        self.length = length
        self.state_dim = state_dim
        self.buffer: list[ReplayMemory._Episode] = []

    def is_filled(self) -> bool:
        return len(self.buffer) >= self.length

    def _get_empty_episode(self):
        if len(self.buffer) <= self.index:
            episode = ReplayMemory._Episode(self.state_dim)
            self.buffer.append(episode)
        else:
            episode = self.buffer[self.index]

        self.index = (self.index + 1) % self.length
        if self.index < len(self.buffer):
            self.buffer[self.index].disable()

        return episode

    def push(self, state, reward, max_q):
        episode = self._get_empty_episode()
        episode.set(state, reward, max_q)

    def push_end_episode(self):
        episode = self._get_empty_episode()
        episode.disable()

    def select_randomly(self):
        import random

        i = random.randrange(0, len(self.buffer) - 1)
        next_i = (i + 1) % self.length
        while not self.buffer[next_i].is_valid:
            i = random.randrange(0, len(self.buffer) - 1)
            next_i = (i + 1) % self.length

        return {
            "state": self.buffer[i].state,
            "reward": self.buffer[i].reward,
            "next_max_q": self.buffer[next_i].max_q,
        }

    def reset(self):
        self.index = 0
        for e in self.buffer:
            e.disable()


class DQN(torch.nn.Module):
    def __init__(self):
        from collections import OrderedDict

        super().__init__()

        self.seq = torch.nn.Sequential(OrderedDict([
            ("body1", torch.nn.Linear(5, 32)),
            ("act1", torch.nn.Tanh()),
            ("body2", torch.nn.Linear(32, 32)),
            ("act2", torch.nn.Tanh()),
            ("body3", torch.nn.Linear(32, 16)),
            ("act3", torch.nn.Tanh()),
            ("body4", torch.nn.Linear(16, 8)),
            ("act4", torch.nn.Tanh()),
            ("body5", torch.nn.Linear(8, 16)),
            ("act5", torch.nn.Tanh()),
            ("body6", torch.nn.Linear(16, 17)),
            ("act6", torch.nn.Tanh()),
        ]))

    def forward(self, x) -> torch.Tensor:
        return self.seq(x)
