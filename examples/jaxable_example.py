import dataclasses

import numpy as np
import mujoco
from mujoco import mjx

import jax
import jax.numpy as jnp
from flax import nnx

from framework.prelude import *
from framework.pheromone import PheromoneField
from framework.utils import GenericTkinterViewer, Timer
from framework.backends.basic_environment import generate_mjspec


class Controller(JaxableController):
    def __init__(self, interval: int, num_robots: int):
        self.timer = Timer(interval)
        self.output = jnp.zeros((num_robots, 2))

    def forward(self, x: jax.Array) -> jax.Array:
        action_map = jnp.array([
            [-1.0, 1.0],  # LEFT
            [1.0, -1.0],  # RIGHT
            [1.0, 1.0],  # FORWARD
            [-0.8, -0.8]  # BACKWARD
        ])
        if self.timer.tick():
            self.output = action_map[x[:, 0]]
        return self.output


@dataclasses.dataclass
class SimulationData:
    rngs: nnx.Rngs

    data: mjx.Data

    robots: BatchedRobots

    pheromone: PheromoneField
    pheromone_cell_pos: jax.Array  # Shape: (num_cells, 2, 2), # [(cell_pos[0], cell_pos[1]), (index_x, index_y))]

    controller: Controller

    @staticmethod
    @jax.jit
    def _jit_step(
            controller_output, pheromone_cell_pos, pheromone, robots, data, model: mjx.Model, dt: float
    ):
        robots.update(data)

        pheromone.update(dt)

        data = robots.set_ctrl(data, controller_output)

        distance = jnp.linalg.norm(
            robots.positions[:, None, :2] - pheromone_cell_pos[None, :, 0, :2],
            axis=2
        )
        closest_indices = jnp.argmin(distance, axis=1)
        pheromone.add_liquid(
            xs=pheromone_cell_pos[closest_indices, 1, 0],
            ys=pheromone_cell_pos[closest_indices, 1, 1],
            additions=1.0
        )

        data = mjx.step(model, data)

        return robots, pheromone, data

    def step(self, model: mjx.Model, dt: float) -> 'SimulationData':
        input_ = jax.random.randint(self.rngs(), shape=(self.robots.num_robots, 1), minval=0, maxval=4)
        controller_output = self.controller.forward(input_)

        updated_robots, updated_pheromone, updated_data = SimulationData._jit_step(
            controller_output, self.pheromone_cell_pos, self.pheromone,
            self.robots, self.data, model, dt
        )

        return SimulationData(
            rngs=self.rngs,
            data=updated_data,
            robots=updated_robots,
            pheromone=updated_pheromone,
            pheromone_cell_pos=self.pheromone_cell_pos,
            controller=self.controller
        )

    def render(
            self,
            mj_model: mujoco.MjModel,
            render_shape: tuple[int, int],
            max_geom: int,
            img_buf: np.ndarray,
            pos: tuple[float, float, float],
            lookat: tuple[float, float, float]
    ):
        from framework.backends.utils import render

        mj_data = mjx.get_data(mj_model, self.data)
        render(mj_model, mj_data, render_shape, max_geom, img_buf, pos, lookat)

    def reset(self, model: mjx.Model):
        self.data = mjx.make_data(model)
        self.pheromone.reset()
        self.robots.update(self.data)

    def get_scores(self) -> list[float]:
        return []

    def calc_total_score(self) -> float:
        return 0.0


class Simulator(SimulatorBackend):
    def __init__(self, settings: Settings):
        mj_spec, nest_spec, robot_specs, food_specs, pheromone_cell_specs = generate_mjspec(settings)

        self.dt = settings.Simulation.TIME_STEP
        self.render_shape = settings.Render.RENDER_WIDTH, settings.Render.RENDER_HEIGHT
        self.max_geom = settings.Render.MAX_GEOM

        self.mj_model = mj_spec.compile()
        self.model = mjx.put_model(self.mj_model)
        data = mjx.put_data(self.mj_model, mujoco.MjData(self.mj_model))

        robot_ids = BatchedRobotIDs.from_specs(self.mj_model, robot_specs)
        robots = BatchedRobots(
            data,
            robot_ids,
            d=settings.Robot.DISTANCE_BETWEEN_WHEELS,
            velocity=settings.Robot.MAX_SPEED
        )

        pheromone_field = PheromoneField(
            nx=settings.Pheromone.WIDTH_NUM,
            ny=settings.Pheromone.HEIGHT_NUM,
            dx=settings.Pheromone.CELL_SIZE,
            material=settings.Pheromone.MATERIAL,
            evaporation_rate=settings.Pheromone.EVAPORATION_RATE,
            decrease_rate=settings.Pheromone.DECREASE_RATE,
            temperature=settings.Simulation.TEMPERATURE,
            iter_=settings.Pheromone.ITERATIONS_PER_STEP,
        )
        self.pheromone_cells = [s.get_cell(self.mj_model) for s in pheromone_cell_specs]
        pheromone_cell_pos = jnp.array(
            [
                ((cell.pos[0], cell.pos[1]), (cell.index_x, cell.index_y))
                for cell in self.pheromone_cells
            ], dtype=np.float32
        )

        self.simulation_data = SimulationData(
            rngs=nnx.Rngs(0),
            data=data,
            robots=robots,
            pheromone=pheromone_field,
            pheromone_cell_pos=pheromone_cell_pos,
            controller=Controller(int(1 / settings.Simulation.TIME_STEP), settings.Robot.NUM)
        )

    def step(self):
        self.simulation_data = self.simulation_data.step(self.model, self.dt)

    def render(self, img_buf: np.ndarray, pos: tuple[float, float, float], lookat: tuple[float, float, float]):
        if img_buf is None:
            return

        color_max = 1.0
        pheromone: jnp.ndarray = self.simulation_data.pheromone.values_gas
        for cell in self.pheromone_cells:
            pheromone_value = float(pheromone[cell.index_y, cell.index_x])
            rgba: tuple[float, float, float] = (pheromone_value / color_max, 0.0, 1 - pheromone_value / color_max)
            cell.set_color(*rgba, 0.5)

        self.simulation_data.render(
            self.mj_model,
            render_shape=self.render_shape,
            max_geom=self.max_geom,
            img_buf=img_buf,
            pos=pos,
            lookat=lookat
        )

    def reset(self):
        self.simulation_data.reset(self.model)

    def get_scores(self) -> list[float]:
        return self.simulation_data.get_scores()

    def calc_total_score(self) -> float:
        return self.simulation_data.calc_total_score()


def jaxable_example():
    settings = Settings()
    settings.Render.RENDER_WIDTH = 480
    settings.Render.RENDER_HEIGHT = 320

    settings.Pheromone.ACTIVE = True

    settings.Robot.NUM = 1
    settings.Food.NUM = 1

    viewer = GenericTkinterViewer(
        settings,
        Simulator(settings),
    )
    viewer.run()


if __name__ == '__main__':
    jaxable_example()
