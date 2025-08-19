import enum

import numpy as np
import mujoco

from framework.prelude import *
from framework.backends import BasicSimulator
from framework.utils import GenericTkinterViewer, Timer


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
    def __init__(self, settings: Settings):
        super().__init__(settings, True)

        self.settings = settings
        self.controller = Controller(int(1 / settings.Simulation.TIME_STEP), settings.Robot.NUM)

        mujoco.mj_step(self.model, self.data)

    def step(self):
        output_ndarray = self.controller.forward(None)
        for i, robot in enumerate(self.robot_values):
            robot.act(
                right_wheel=output_ndarray[i, 0],
                left_wheel=output_ndarray[i, 1]
            )

        self.add_pheromone(
            np.array([r.xpos for r in self.robot_values]),
            np.array([1.0])
        )

        self._pheromone_field.add_liquid_by_cell(self._pheromone_cells)
        self._pheromone_field.update(self.settings.Simulation.TIME_STEP)
        mujoco.mj_step(self.model, self.data)

    def get_scores(self) -> list[float]:
        return []

    def calc_total_score(self) -> float:
        return 0.0


def viewer_example():
    settings = Settings()
    settings.Render.RENDER_WIDTH = 480
    settings.Render.RENDER_HEIGHT = 320

    settings.Pheromone.ACTIVE = True

    settings.Food.NUM = 0

    backend = Simulator(settings)
    viewer = GenericTkinterViewer(settings, backend)
    viewer.run()


if __name__ == '__main__':
    viewer_example()
