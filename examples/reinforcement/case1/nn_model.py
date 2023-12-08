import torch


class ReplayMemory:
    class _Episode:
        def __init__(self, state_dim):
            self.is_valid = False
            self.state = torch.Tensor([0.0] * state_dim)
            self.reward = 0.0

        def set(self, state, action_id, reward: float):
            self.is_valid = True
            self.state[:] = state
            self.action_id = action_id
            self.reward = reward

        def disable(self):
            self.is_valid = False

    def __init__(self, state_dim: int, length: int):
        self.index = 0
        self.length = length
        self.state_dim = state_dim
        self.buffer: list[ReplayMemory._Episode] = []

    def is_filled(self) -> bool:
        return len(self.buffer) >= self.length

    def _push_empty_episode(self):
        if not self.is_filled():
            episode = ReplayMemory._Episode(self.state_dim)
            self.buffer.append(episode)
        else:
            episode = self.buffer[self.index]

        self.index = (self.index + 1) % self.length
        if self.index < len(self.buffer):
            self.buffer[self.index].disable()

        return episode

    def _get_last_episode(self):
        if len(self.buffer) == 0:
            raise "This replay buffer is empty."

        if self.index == 0:
            return self.buffer[len(self.buffer) - 1]
        return self.buffer[self.index - 1]

    def push(self, state, action_id):
        episode = self._push_empty_episode()
        episode.set(state, action_id, 0.0)

    def push_end_episode(self):
        episode = self._push_empty_episode()
        episode.disable()

    def set_reward(self, reward):
        if len(self.buffer) == 0:
            return
        episode = self._get_last_episode()
        episode.reward = reward

    def select_randomly(self):
        import random

        i = random.randrange(0, len(self.buffer) - 1)
        next_i = (i + 1) % self.length
        while not (self.buffer[i].is_valid and self.buffer[next_i].is_valid):
            i = random.randrange(0, len(self.buffer) - 1)
            next_i = (i + 1) % self.length

        return {
            "state": self.buffer[i].state,
            "action": self.buffer[i].action_id,
            "reward": self.buffer[i].reward,
            "next_state": self.buffer[i].state,
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
            ("body1", torch.nn.Linear(6, 128)),
            ("act1", torch.nn.Tanh()),
            ("body2", torch.nn.Linear(128, 64)),
            ("act2", torch.nn.Tanh()),
            ("body3", torch.nn.Linear(64, 32)),
            ("act3", torch.nn.Tanh()),
            ("body4", torch.nn.Linear(32, 17)),
        ]))

    def forward(self, x) -> torch.Tensor:
        return self.seq(x)
