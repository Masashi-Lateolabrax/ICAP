import numpy as np

from framework.backends import MujocoBackend
from framework.utils import GenericTkinterViewer
from framework.prelude import Settings, RobotLocation, Position


class SampleMujocoBackend(MujocoBackend):
    def __init__(self, settings: Settings, render: bool = False):
        super().__init__(settings, render)

    def step(self):
        super().step()


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
