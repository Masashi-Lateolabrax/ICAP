import torch

from ..property import RobotProperty
from .omni_sensor import OmniSensor


class RobotInput:
    def __init__(self, property_: RobotProperty, other_robot_sensor: OmniSensor, food_sensor: OmniSensor):
        self.property = property_
        self.robot_sensor = other_robot_sensor
        self.food_sensor = food_sensor
        self.touch = torch.zeros(4, dtype=torch.float32)

    def get(self) -> torch.Tensor:
        self.touch[0:2] = torch.tensor(self.robot_sensor.get(self.property.global_direction, self.property.pos)[0:2])
        self.touch[2:4] = torch.tensor(self.food_sensor.get(self.property.global_direction, self.property.pos)[0:2])
        return self.touch

    def get_food(self) -> torch.Tensor:
        return self.get()[2:4]
