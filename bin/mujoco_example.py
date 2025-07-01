import mujoco
import numpy as np

from framework.backends import MujocoBackend
from framework.utils import GenericTkinterViewer
from framework.sensor import OmniSensor, DirectionSensor
from framework.prelude import Settings, RobotLocation, Position
from framework.sensor import PreprocessedOmniSensor, DirectionSensor
from framework.prelude import Settings, RobotLocation, Position, SensorInterface



class Controller:
    def __init__(self, sensors: list[tuple[OmniSensor, OmniSensor, DirectionSensor]]):
        self.sensors = sensors


class SampleMujocoBackend(MujocoBackend):
    def __init__(self, settings: Settings, render: bool = False):
        super().__init__(settings, render)
        self.scores = []

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

        self.controller = Controller(sensors)

    def step(self):
        for robot in self.robot_values:
            robot_pos = robot._center_site.xpos[0:2]
            xdirection = robot._front_site.xpos[0:2] - robot_pos
            xdirection /= np.linalg.norm(xdirection)

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
