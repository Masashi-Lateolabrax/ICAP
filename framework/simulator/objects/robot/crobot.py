import numpy as np

from ....interfaceis import CBrainInterface

from .robot_property import RobotProperty
from .robot_input import RobotInput
from .actuator import Actuator


class CRobot:
    def __init__(
            self,
            brain: CBrainInterface,
            property_: RobotProperty,
            input_: RobotInput,
            actuator: Actuator,
    ):
        self.brain = brain
        self.property = property_
        self.input = input_
        self.actuator = actuator

    def calc_relative_position(self, target_pos):
        """
        Calculate relative position from robot to target_pos

        Args:
            target_pos: target position. [x, y, z]

        Returns:
            relative position. [front, left, up]
        """
        direction_to_front = self.property.global_direction[:2]
        direction_to_left = np.array([direction_to_front[1], -direction_to_front[0]], dtype=np.float64)
        front = np.dot(target_pos[:2] - self.property.pos[:2], direction_to_front)
        left = np.dot(target_pos[:2] - self.property.pos[:2], direction_to_left)
        return np.array([left, front, 0], dtype=np.float64)

    def update(self):
        self.property.update()

    @property
    def size(self):
        return self.property.size

    @property
    def position(self):
        return self.property.pos

    @property
    def local_direction(self):
        return self.property.local_direction

    @property
    def global_direction(self):
        return self.property.global_direction

    @property
    def angle(self):
        return self.property.angle

    def action(self, _input=None):
        out = self.brain.think(self.input)
        self.actuator.execute(out)
