import numpy as np
import mujoco
from icecream import ic

from framework.prelude import *
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


class Simulator(SimulatorBackend):
    def __init__(self, settings):
        self.settings = settings
        self.render_shape = self.settings.Render.RENDER_WIDTH, self.settings.Render.RENDER_HEIGHT

        spec, pheromone_site_spec = _generate_mjspec(self.settings)

        self.spec: mujoco.MjSpec = spec
        self.model: mujoco.MjModel = self.spec.compile()
        self.data: mujoco.MjData = mujoco.MjData(self.model)

        self.pheromone_cells: list[PheromoneFieldCell] = [s.get_cell(self.model) for s in pheromone_site_spec]
        self.pheromone_field: PheromoneField = PheromoneField(
            nx=self.settings.Pheromone.WIDTH_NUM,
            ny=self.settings.Pheromone.HEIGHT_NUM,
            dx=self.settings.Pheromone.CELL_SIZE,
            material=self.settings.Pheromone.MATERIAL,
            diffusion_coefficient=self.settings.Pheromone.DIFFUSION_COEFFICIENT,
            evaporation_rate=self.settings.Pheromone.EVAPORATION_RATE,
            decrease_rate=self.settings.Pheromone.DECREASE_RATE,
            temperature=self.settings.Pheromone.TEMPERATURE,
            iter_=self.settings.Pheromone.ITERATIONS_PER_STEP,
        )

        self.camera = mujoco.MjvCamera()

    def step(self):
        [c for c in self.pheromone_cells if c.index_x == 5 and c.index_y == 5][0].add_value = 1.0

        self.pheromone_field.add_liquid_by_cell(self.pheromone_cells)
        self.pheromone_field.update(self.settings.Simulation.TIME_STEP)
        mujoco.mj_step(self.model, self.data)

    def render(self, img_buf: np.ndarray, pos: tuple[float, float, float], lookat: tuple[float, float, float]):
        if img_buf is None:
            return

        color_max = 1.0
        pheromone: np.ndarray = self.pheromone_field.get_gas_all()
        print(pheromone)
        for i, cell in enumerate(self.pheromone_cells):
            pheromone_value = float(pheromone[cell.index_y, cell.index_x])
            rgba: tuple[float, float, float] = (pheromone_value / color_max, 0.0, 1 - pheromone_value / color_max)
            cell.set_color(*rgba, 0.5)

        try:
            pos = np.array(pos)
            lookat = np.array(lookat)
            sub = pos - lookat
            self.camera.lookat[:] = lookat
            self.camera.distance = np.linalg.norm(sub)
            self.camera.azimuth = np.arctan2(
                sub[1], sub[0]
            ) * 180 / mujoco.mjPI + 180
            self.camera.elevation = -np.arcsin(
                sub[2] / self.camera.distance
            ) * 180 / mujoco.mjPI

            with mujoco.Renderer(self.model, width=self.render_shape[0], height=self.render_shape[1]) as renderer:
                renderer.update_scene(self.data, self.camera)
                renderer.render(out=img_buf)

        except Exception as e:
            ic("MuJoCo render error:", e)
            img_buf.fill(0)

    def reset(self):
        raise NotImplementedError

    def get_scores(self) -> list[float]:
        raise NotImplementedError

    def calc_total_score(self) -> float:
        raise NotImplementedError


def viewer_example():
    settings = Settings()
    settings.Render.RENDER_WIDTH = 480
    settings.Render.RENDER_HEIGHT = 320

    backend = Simulator(settings)
    viewer = GenericTkinterViewer(settings, backend)
    viewer.run()


if __name__ == '__main__':
    viewer_example()
