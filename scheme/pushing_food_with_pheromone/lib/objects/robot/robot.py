import torch

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
        self.touch[0:2] = torch.tensor(self.robot_sensor.get(self.data.direction, self.data.pos)[0:2])
        self.touch[2:4] = torch.tensor(self.food_sensor.get(self.data.direction, self.data.pos)[0:2])
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

    def update(self):
        self._data.update()

    @property
    def position(self):
        return self._data.pos

    @property
    def direction(self):
        return self._data.direction

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
