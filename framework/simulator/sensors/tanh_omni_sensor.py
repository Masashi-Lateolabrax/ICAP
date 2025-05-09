import numpy as np
from mujoco._structs import _MjDataSiteViews

from .omni_sensor import OmniSensor


class TanhOmniSensor(OmniSensor):
    def __init__(self, gain: float, offset: float, target_sites: list[_MjDataSiteViews], tanh_gain: float = 1):
        super().__init__(gain, offset, target_sites)
        self._tanh_gain = tanh_gain

    def get(self, bot_direction, bot_pos) -> np.ndarray:
        raw = super().get(bot_direction, bot_pos)
        return np.tanh(raw * self._tanh_gain)
