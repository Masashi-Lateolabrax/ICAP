import numpy as np
import torch

from .brain import BrainInterface, BrainJudgement, DiscreteOutput
from .robot_property import RobotProperty
from .robot_input import RobotInput
from .actuator import Actuator


class Robot:
    def __init__(
            self,
            name: str,
            brain: BrainInterface,
            property_: RobotProperty,
            input_: RobotInput,
            actuator: Actuator,
            center_site_rgba: np.ndarray,
    ):
        self.name = name
        self.brain = brain
        self.property = property_
        self.input = input_
        self.actuator = actuator
        self.center_site_rgba = center_site_rgba

    def set_color(self, r, g, b, a):
        self.center_site_rgba[0:4] = [r, g, b, a]

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

    def think(self) -> torch.Tensor:
        return self.brain.think(self.input)

    def action(self, output: torch.Tensor):
        if isinstance(self.brain, DiscreteOutput):
            output = self.brain.convert(output)

        if isinstance(output, torch.Tensor):
            self.actuator.execute(output)

        elif isinstance(output, BrainJudgement):
            match output:
                case BrainJudgement.STOP:
                    self.actuator.stop()
                case BrainJudgement.FORWARD:
                    self.actuator.forward()
                case BrainJudgement.BACK:
                    self.actuator.back()
                case BrainJudgement.TURN_RIGHT:
                    self.actuator.turn_right()
                case BrainJudgement.TURN_LEFT:
                    self.actuator.turn_left()
                case BrainJudgement.SECRETION:
                    self.actuator.secretion()
                case _:
                    raise ValueError("Invalid judge")

        else:
            raise TypeError(f"Invalid output type. Type: {type(output)}")

    def exec(self):
        output = self.think()
        self.action(output)
