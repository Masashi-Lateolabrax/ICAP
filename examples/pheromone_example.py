import numpy as np
import mujoco

from framework.prelude import *
from framework.backends import BasicMuJoCoSimulator
from framework.utils import GenericTkinterViewer
from framework.environment import setup_option, setup_visual, setup_textures, add_geom
from framework.pheromone import add_pheromone_cell, PheromoneFieldCellSpec, PheromoneFieldCell, PheromoneField


def _generate_mjspec(
        settings: Settings
) -> tuple[mujoco.MjSpec, list[PheromoneFieldCellSpec]]:
    spec = mujoco.MjSpec()

    setup_option(spec, settings)
    setup_visual(spec, settings)
    setup_textures(spec, settings)

    add_geom(
        spec.worldbody,
        geom_type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=(0, 0, 0),
        size=(
            settings.Simulation.WORLD_WIDTH * 0.5,
            settings.Simulation.WORLD_HEIGHT * 0.5,
            1
        ),
        material="ground",
        rgba=GROUND_COLOR,
        condim=GROUND_COLLISION_CONDIM
    )

    sites = []
    for x in range(settings.Pheromone.WIDTH_NUM):
        for y in range(settings.Pheromone.HEIGHT_NUM):
            pos_x = settings.Pheromone.CELL_SIZE * (x - (settings.Pheromone.WIDTH_NUM - 1) * 0.5)
            pos_y = settings.Pheromone.CELL_SIZE * (-y + (settings.Pheromone.HEIGHT_NUM - 1) * 0.5)

            sites.append(
                add_pheromone_cell(
                    spec,
                    index_x=x,
                    index_y=y,
                    size=settings.Pheromone.CELL_SIZE * 0.5,
                    pos=(pos_x, pos_y, 0),
                )
            )

    return spec, sites


class Simulator(BasicMuJoCoSimulator):
    def __init__(self, settings: Settings):
        spec, pheromone_site_spec = _generate_mjspec(settings)
        super().__init__(settings, spec, render=True)

        self.settings = settings

        self.pheromone_cells: list[PheromoneFieldCell] = [s.get_cell(self.model) for s in pheromone_site_spec]
        self.pheromone_field: PheromoneField = PheromoneField(
            nx=settings.Pheromone.WIDTH_NUM,
            ny=settings.Pheromone.HEIGHT_NUM,
            dx=settings.Pheromone.CELL_SIZE,
            material=settings.Pheromone.MATERIAL,
            diffusion_coefficient=settings.Pheromone.DIFFUSION_COEFFICIENT,
            evaporation_rate=settings.Pheromone.EVAPORATION_RATE,
            decrease_rate=settings.Pheromone.DECREASE_RATE,
            temperature=settings.Pheromone.TEMPERATURE,
            iter_=settings.Pheromone.ITERATIONS_PER_STEP,
        )

    def step(self):
        center_cell = sorted(
            [(c, np.linalg.norm(c.pos)) for i, c in enumerate(self.pheromone_cells)],
            key=lambda x: x[1]
        )[0][0]
        center_cell.add_value = 1.0

        self.pheromone_field.add_liquid_by_cell(self.pheromone_cells)
        self.pheromone_field.update(self.settings.Simulation.TIME_STEP)
        mujoco.mj_step(self.model, self.data)

    def render(self, img_buf: np.ndarray, pos: tuple[float, float, float], lookat: tuple[float, float, float]):
        color_max = 1.0
        pheromone: np.ndarray = self.pheromone_field.get_gas_all()
        for i, cell in enumerate(self.pheromone_cells):
            pheromone_value = float(pheromone[cell.index_y, cell.index_x])
            rgba: tuple[float, float, float] = (pheromone_value / color_max, 0.0, 1 - pheromone_value / color_max)
            cell.set_color(*rgba, 0.5)

        super().render(img_buf, pos, lookat)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.pheromone_field.reset()

    def get_scores(self) -> list[float]:
        return []

    def calc_total_score(self) -> float:
        return 0.0


def viewer_example():
    settings = Settings()
    settings.Render.RENDER_WIDTH = 480
    settings.Render.RENDER_HEIGHT = 320

    backend = Simulator(settings)
    viewer = GenericTkinterViewer(settings, backend)
    viewer.run()


if __name__ == '__main__':
    viewer_example()
