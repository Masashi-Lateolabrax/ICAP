import numpy as np

from mujoco._structs import _MjDataSiteViews

from ._omni_sensor import OmniSensor
from ..prelude import *


class PreprocessedOmniSensor(OmniSensor):
    def __init__(
            self,
            robot: RobotValues,
            d_gain: float,
            offset: float,
            target_sites: list[_MjDataSiteViews]
    ):
        super().__init__(robot, 1, d_gain, offset, target_sites)

    def get(self) -> np.ndarray:
        raw = super().get()
        magnitude = 1 / (np.linalg.norm(raw) + 1)
        angle = np.arctan2(-raw[0], raw[1]) / np.pi
        return np.array([magnitude, angle])
