import numpy as np
import mujoco

from framework.prelude import *
from framework.backends import BasicSimulator
from framework.utils import GenericTkinterViewer


class Simulator(BasicSimulator):
    def __init__(self, settings: Settings):
        super().__init__(settings, True)

        self.settings = settings

        mujoco.mj_step(self.model, self.data)

    def step(self):
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

    backend = Simulator(settings)
    viewer = GenericTkinterViewer(settings, backend)
    viewer.run()


if __name__ == '__main__':
    viewer_example()
