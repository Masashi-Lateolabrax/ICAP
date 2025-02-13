import unittest
from unittest.mock import Mock

import mujoco
import torch
import numpy as np

from scheme.pushing_food_with_pheromone.lib.parts import OmniSensor, BrainJudgement, BrainInterface

from .robot_data import RobotData
from .actuator import Actuator


class BrainInput:
    def __init__(self, data: RobotData, other_robot_sensor: OmniSensor, food_sensor: OmniSensor):
        self.data = data
        self.robot_sensor = other_robot_sensor
        self.food_sensor = food_sensor
        self.touch = torch.zeros(4, dtype=torch.float32)

    def get(self) -> torch.Tensor:
        self.touch[0:2] = torch.tensor(self.robot_sensor.get(self.data.global_direction, self.data.pos)[0:2])
        self.touch[2:4] = torch.tensor(self.food_sensor.get(self.data.global_direction, self.data.pos)[0:2])
        return self.touch

    def get_food(self) -> torch.Tensor:
        return self.get()[2:4]


class Robot:
    def __init__(
            self,
            brain: BrainInterface,
            body_,
            data: RobotData,
            actuator: Actuator,
            other_robot_sensor: OmniSensor,
            food_sensor: OmniSensor,
    ):
        self.brain = brain
        self.body = body_
        self._data = data
        self._actuator = actuator

        self._input = BrainInput(data, other_robot_sensor, food_sensor)

    def calc_relative_position(self, target_pos):
        """
        Calculate relative position from robot to target_pos

        Args:
            target_pos: target position. [x, y, z]

        Returns:
            relative position. [front, left, up]
        """
        direction_to_front = self._data.global_direction[:2]
        direction_to_left = np.array([direction_to_front[1], -direction_to_front[0]], dtype=np.float64)
        front = np.dot(target_pos[:2] - self._data.pos[:2], direction_to_front)
        left = np.dot(target_pos[:2] - self._data.pos[:2], direction_to_left)
        return np.array([left, front, 0], dtype=np.float64)

    def update(self):
        self._data.update()

    @property
    def position(self):
        return self._data.pos

    @property
    def local_direction(self):
        return self._data.local_direction

    @property
    def global_direction(self):
        return self._data.global_direction

    @property
    def angle(self):
        return self._data.angle

    def action(self, _input=None):
        match self.brain.think(self._input):
            case BrainJudgement.STOP:
                self._actuator.stop()
            case BrainJudgement.FORWARD:
                self._actuator.forward()
            case BrainJudgement.BACK:
                self._actuator.back()
            case BrainJudgement.TURN_RIGHT:
                self._actuator.turn_right()
            case BrainJudgement.TURN_LEFT:
                self._actuator.turn_left()
            case BrainJudgement.SECRETION:
                self._actuator.secretion()
            case _:  # pragma: no cover
                raise ValueError("Invalid judge")


class _TestRobot(unittest.TestCase):
    @staticmethod
    def calc_quat(angle):
        res = np.zeros(4, dtype=np.float64)
        mujoco.mju_axisAngle2Quat(res, [0, 0, 1], angle)
        return res

    def test_calc_relative_position_with_zero_target(self):
        robot = Robot(
            brain=Mock(),
            body_=Mock(),
            data=RobotData(Mock(), Mock(), Mock()),
            actuator=Mock(),
            other_robot_sensor=Mock(),
            food_sensor=Mock(),
        )
        robot._data.angle = 0
        robot._data.xquat = self.calc_quat(np.pi / 2)
        robot._data.pos = np.array([0, 0, 0], dtype=np.float64)
        target_pos = np.array([0, 0, 0], dtype=np.float64)
        result = robot.calc_relative_position(target_pos)
        np.testing.assert_array_equal(result, np.array([0, 0, 0], dtype=np.float64))

    def test_calc_relative_position_case1(self):
        robot = Robot(
            brain=Mock(),
            body_=Mock(),
            data=RobotData(Mock(), Mock(), Mock()),
            actuator=Mock(),
            other_robot_sensor=Mock(),
            food_sensor=Mock(),
        )

        robot._data.angle = 0
        robot._data.xquat = self.calc_quat(0)
        robot._data.pos = np.array([1, 1, 0], dtype=np.float64)
        result = robot.calc_relative_position(
            np.array([2, 2, 0], dtype=np.float64)
        )
        np.testing.assert_array_equal(result, np.array([1, 1, 0], dtype=np.float64))

    def test_calc_relative_position_case2(self):
        robot = Robot(
            brain=Mock(),
            body_=Mock(),
            data=RobotData(Mock(), Mock(), Mock()),
            actuator=Mock(),
            other_robot_sensor=Mock(),
            food_sensor=Mock(),
        )

        robot._data.angle = 0
        robot._data.xquat = self.calc_quat(np.pi / 2)
        robot._data.pos = np.array([1, 1, 0], dtype=np.float64)
        result = robot.calc_relative_position(
            np.array([2, 2, 0], dtype=np.float64)
        )

        assert np.allclose(result, np.array([1, -1, 0], dtype=np.float64))

    def test_calc_relative_position_case3(self):
        robot = Robot(
            brain=Mock(),
            body_=Mock(),
            data=RobotData(Mock(), Mock(), Mock()),
            actuator=Mock(),
            other_robot_sensor=Mock(),
            food_sensor=Mock(),
        )

        robot._data.angle = 0
        robot._data.xquat = self.calc_quat(np.pi)
        robot._data.pos = np.array([1, 1, 0], dtype=np.float64)
        result = robot.calc_relative_position(
            np.array([0, 0, 0], dtype=np.float64)
        )

        assert np.allclose(result, np.array([1, 1, 0], dtype=np.float64))

    def test_calc_relative_position_case4(self):
        robot = Robot(
            brain=Mock(),
            body_=Mock(),
            data=RobotData(Mock(), Mock(), Mock()),
            actuator=Mock(),
            other_robot_sensor=Mock(),
            food_sensor=Mock(),
        )

        robot._data.angle = 0
        robot._data.xquat = self.calc_quat(3 * np.pi / 2)
        robot._data.pos = np.array([1, 1, 0], dtype=np.float64)
        result = robot.calc_relative_position(
            np.array([0, 0, 0], dtype=np.float64)
        )

        assert np.allclose(result, np.array([1, -1, 0], dtype=np.float64))
