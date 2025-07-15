import mujoco
from mujoco._structs import (_MjDataSiteViews, _MjDataJointViews, _MjDataSensorViews)
import numpy as np

from .position import Position


class FoodSpec:
    def __init__(
            self,
            center_site: mujoco._specs.MjsSite,
            free_joint: mujoco._specs.MjsJoint,
            velocimeter: mujoco._specs.MjsSensor
    ):
        self.center_site = center_site
        self.free_joint = free_joint
        self.velocimeter = velocimeter


class FoodValues:
    def __init__(self, data: mujoco.MjData, spec: FoodSpec):
        self._center_site: _MjDataSiteViews = data.site(spec.center_site.name)
        self.joint: _MjDataJointViews = data.joint(spec.free_joint.name)
        self._velocimeter: _MjDataSensorViews = data.sensor(spec.velocimeter.name)

    @property
    def site(self):
        return self._center_site

    @property
    def xpos(self):
        return self._center_site.xpos[0:2]

    @property
    def position(self) -> Position:
        return Position(self.xpos[0], self.xpos[1])


class DummyFoodValues:
    """
    A dummy food values class that preserves evaluation continuity 
    when real food is respawned. Stores frozen position data.
    """

    def __init__(self, original_food_values: FoodValues):
        """
        Create a dummy food values instance that preserves the position
        of the original food at the moment of respawning.
        
        Args:
            original_food_values: The original FoodValues instance to preserve
        """
        # Store the position at the moment of respawning
        self._frozen_xpos = original_food_values.xpos.copy()

    @property
    def xpos(self):
        """Return the frozen position from when the food was respawned"""
        return self._frozen_xpos
