import unittest
from unittest.mock import Mock

import mujoco
import numpy as np

from .brain import BrainInterface, BrainJudgement
from .property import RobotProperty
from .input import RobotInput
from .actuator import Actuator


class Robot:
    def __init__(
            self,
            brain: BrainInterface,
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
        match self.brain.think(self.input):
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
            case _:  # pragma: no cover
                raise ValueError("Invalid judge")


class _TestRobot(unittest.TestCase):
    @staticmethod
    def calc_quat(angle):
        res = np.zeros(4, dtype=np.float64)
        mujoco.mju_axisAngle2Quat(res, [0, 0, 1], angle)
        return res

    @staticmethod
    def create_robot(xquat, pos):
        property_ = RobotProperty(Mock(), Mock(), Mock())
        property_.angle = 0
        property_.xquat = xquat
        property_.pos = pos
        return Robot(
            brain=Mock(),
            property_=property_,
            input_=Mock(),
            actuator=Mock(),
        )

    def test_calc_relative_position_with_zero_target(self):
        robot = self.create_robot(
            self.calc_quat(0),
            np.array([0, 0, 0], dtype=np.float64)
        )
        target_pos = np.array([0, 0, 0], dtype=np.float64)
        result = robot.calc_relative_position(target_pos)
        assert np.allclose(result, np.array([0, 0, 0], dtype=np.float64))

    def test_calc_relative_position_case1(self):
        robot = self.create_robot(
            self.calc_quat(0),
            np.array([1, 1, 0], dtype=np.float64)
        )
        result = robot.calc_relative_position(
            np.array([2, 2, 0], dtype=np.float64)
        )
        assert np.allclose(result, np.array([1, 1, 0], dtype=np.float64))

    def test_calc_relative_position_case2(self):
        robot = self.create_robot(
            self.calc_quat(np.pi / 2),
            np.array([1, 1, 0], dtype=np.float64)
        )
        result = robot.calc_relative_position(
            np.array([2, 2, 0], dtype=np.float64)
        )
        assert np.allclose(result, np.array([1, -1, 0], dtype=np.float64))

    def test_calc_relative_position_case3(self):
        robot = self.create_robot(
            self.calc_quat(np.pi),
            np.array([1, 1, 0], dtype=np.float64)
        )
        result = robot.calc_relative_position(
            np.array([0, 0, 0], dtype=np.float64)
        )
        assert np.allclose(result, np.array([1, 1, 0], dtype=np.float64))

    def test_calc_relative_position_case4(self):
        robot = self.create_robot(
            self.calc_quat(3 * np.pi / 2),
            np.array([1, 1, 0], dtype=np.float64)
        )
        result = robot.calc_relative_position(
            np.array([0, 0, 0], dtype=np.float64)
        )
        assert np.allclose(result, np.array([1, -1, 0], dtype=np.float64))
