import numpy as np
from mujoco._structs import _MjDataSiteViews


class OmniSensor:
    def __init__(self, gain: float, offset: float, target_sites: list[_MjDataSiteViews]):
        self.gain = gain
        self.offset = offset
        self.targets: list[_MjDataSiteViews] = target_sites

    def _update_positions(self):
        self.target_positions = np.zeros((len(self.targets), 2))
        for i, p in enumerate(self.targets):
            self.target_positions[i] = p.xpos[0:2]

    def get(self, bot_direction, bot_pos):
        direction = np.array([
            [bot_direction[1], -bot_direction[0]],
            [bot_direction[0], bot_direction[1]],
        ])

        sub = self.target_positions - bot_pos[0:2]
        distance = np.linalg.norm(sub, axis=1)
        sub = sub.T / distance
        trigono_components = np.dot(direction, sub)
        scaled_distance = self.gain * np.maximum(distance - self.offset, 0)
        res = np.dot(
            trigono_components,
            np.reciprocal(scaled_distance + 1),
        )
        res /= self.target_positions.shape[0]

        return res
