import torch

from framework.simulator.sensors import OmniSensor, DirectionSensor
from .robot_property import RobotProperty


class RobotInput:
    def __init__(
            self,
            property_: RobotProperty,
            other_robot_sensor: OmniSensor,
            food_sensor: OmniSensor,
            nest_sensor: DirectionSensor
    ):
        self.property = property_
        self.robot_sensor = other_robot_sensor
        self.food_sensor = food_sensor
        self.nest_sensor = nest_sensor
        self.torch = torch.zeros(6, dtype=torch.float32, requires_grad=False)

    def get(self) -> torch.Tensor:
        self.torch[0:2] = torch.tensor(self.robot_sensor.get(self.property.global_direction, self.property.pos))
        self.torch[2:4] = torch.tensor(self.food_sensor.get(self.property.global_direction, self.property.pos))
        self.torch[4:6] = torch.tensor(self.nest_sensor.get(self.property.global_direction[:2], self.property.pos[:2]))
        return self.torch

    def get_food(self) -> torch.Tensor:
        return self.get()[2:4]
