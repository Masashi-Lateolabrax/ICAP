import numpy as np
from mujoco._structs import _MjDataSiteViews

from .omni_sensor import OmniSensor


class AveOmniSensor(OmniSensor):
    def __init__(self, gain: float, offset: float, target_sites: list[_MjDataSiteViews]):
        super().__init__(gain, offset, target_sites)

    def get(self, bot_direction, bot_pos) -> np.ndarray:
        raw = super().get(bot_direction, bot_pos)
        raw /= len(self.targets)
        return raw
