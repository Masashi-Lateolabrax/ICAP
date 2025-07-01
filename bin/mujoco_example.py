import mujoco
import numpy as np
import torch

from framework.backends import MujocoBackend
from framework.utils import GenericTkinterViewer
from framework.sensor import PreprocessedOmniSensor, DirectionSensor
from framework.controller import Controller
from framework.prelude import Settings, RobotLocation, Position, SensorInterface


class RobotNeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(RobotNeuralNetwork, self).__init__()
        self.linear1 = torch.nn.Linear(6, 3)
        self.activation1 = torch.nn.Tanhshrink()

        self.linear2 = torch.nn.Linear(3, 2)
        self.activation2 = torch.nn.Tanh()

    def forward(self, input_):
        x = self.linear1.forward(input_)
        x = self.activation1.forward(x)
        x = self.linear2.forward(x)
        x = self.activation2.forward(x)
        return x


class SampleMujocoBackend(MujocoBackend):
    def __init__(self, settings: Settings, render: bool = False):
        super().__init__(settings, render)
        self.scores = []
        self.controller = Controller(settings, 32, RobotNeuralNetwork())

        self.sensors: list[tuple[SensorInterface]] = [
            (
                PreprocessedOmniSensor(
                    robot,
                    settings.Robot.ROBOT_SENSOR_GAIN,
                    settings.Robot.RADIUS * 2,
                    [other.site for j, other in enumerate(self.robot_values) if j != i]
                ),
                PreprocessedOmniSensor(
                    robot,
                    settings.Robot.FOOD_SENSOR_GAIN,
                    settings.Robot.RADIUS + settings.Food.RADIUS,
                    [food.site for food in self.food_values]
                ),
                DirectionSensor(
                    robot, self.nest_site, settings.Nest.RADIUS
                )
            )
            for i, robot in enumerate(self.robot_values)
        ]

        self._pairs = [self.controller.register(r, f, n) for r, f, n in self.sensors]

    def __del__(self):
        for p in self._pairs:
            p.finished = True

    def step(self):
        for pair in self._pairs:
            pair.need_calculation = True

        self.controller.calculate()

        for r, p in zip(self.robot_values, self._pairs):
            right_wheel = p.output_buf[0]
            left_wheel = p.output_buf[1]
            r.act(right_wheel, left_wheel)

        mujoco.mj_step(self.model, self.data)

    def score(self) -> list[float]:
        return self.scores


def mujoco_example():
    settings = Settings()
    settings.Render.RENDER_WIDTH = 480
    settings.Render.RENDER_HEIGHT = 320

    settings.Robot.INITIAL_POSITION = [
        RobotLocation(0, 0, np.pi / 2),
    ]
    settings.Food.INITIAL_POSITION = [
        Position(0, 2),
    ]

    backend = SampleMujocoBackend(settings, render=True)
    viewer = GenericTkinterViewer(
        settings,
        backend,
    )
    viewer.run()


if __name__ == '__main__':
    mujoco_example()
