import enum

import mujoco
import numpy as np

from framework.prelude import *
from framework.utils import GenericTkinterViewer
from framework.backends import BasicSimulator
from framework.utils import Timer


class Action(enum.Enum):
    LEFT = 0
    RIGHT = 1
    FORWARD = 2
    BACKWARD = 3


class Controller:
    def __init__(self, interval: int, num_robots: int):
        self.timer = Timer(interval)
        self.output = np.zeros((num_robots, 2))

    def forward(self, _input):
        if self.timer.tick():
            for i in range(self.output.shape[0]):
                action = np.random.choice(list(Action))
                match action:
                    case Action.LEFT:
                        self.output[i, 0] = -1.0
                        self.output[i, 1] = 1.0
                    case Action.RIGHT:
                        self.output[i, 0] = 1.0
                        self.output[i, 1] = -1.0
                    case Action.FORWARD:
                        self.output[i, 0] = 1.0
                        self.output[i, 1] = 1.0
                    case Action.BACKWARD:
                        self.output[i, 0] = -0.8
                        self.output[i, 1] = -0.8

        return self.output


class Simulator(BasicSimulator):
    def __init__(self, settings: Settings, render: bool = False):
        super().__init__(settings, render)
        self.controller = Controller(int(1 / settings.Simulation.TIME_STEP), settings.Robot.NUM)
        mujoco.mj_step(self.model, self.data)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

    def get_scores(self) -> list[float]:
        return []

    def calc_total_score(self) -> float:
        return 0.0

    def step(self):
        output_ndarray = self.controller.forward(None)

        for i, robot in enumerate(self.robot_values):
            robot.act(
                right_wheel=output_ndarray[i, 0],
                left_wheel=output_ndarray[i, 1]
            )

        mujoco.mj_step(self.model, self.data)


def mujoco_example():
    settings = Settings()
    settings.Render.RENDER_WIDTH = 480
    settings.Render.RENDER_HEIGHT = 320

    settings.Robot.NUM = 3
    settings.Food.NUM = 0

    viewer = GenericTkinterViewer(
        settings,
        Simulator(settings, render=True),
    )
    viewer.run()


if __name__ == '__main__':
    mujoco_example()
