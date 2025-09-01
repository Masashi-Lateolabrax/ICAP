import numpy as np
import mujoco
from mujoco import mjx

import jax
import jax.numpy as jnp
from flax import nnx
from flax.struct import dataclass as jax_dataclass

from framework.prelude import *
from framework.utils import GenericTkinterViewer
from framework.backends import BasicSimulatorWithEnv


class Controller(nnx.Module):
    def __init__(self, num_robots: int):
        self.output = jnp.ones((num_robots,))

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.output

    @staticmethod
    def dim():
        return 0

class SimulationData:
    @staticmethod
    def __step(
            model_: mjx.Model,
            data: mjx.Data,
            pheromone: PheromoneField,
            robots: BatchedRobots,
            controller_output: jax.Array,

            time_step: float,
            num_robots: int,
            pheromone_cell_ind: jax.Array,
            pheromone_cell_pos: jax.Array,
    ):
        robots.update(data)

        data = robots.set_ctrl(data, controller_output)

        distance = jnp.linalg.norm(
            robots.positions[:, None, :2] - pheromone_cell_pos[None, :, :2],
            axis=2
        )
        closest_indices = jnp.argmin(distance, axis=1)
        pheromone.add_liquid(
            xs=pheromone_cell_ind[closest_indices, 0],
            ys=pheromone_cell_ind[closest_indices, 1],
            additions=jnp.ones((num_robots,), dtype=jnp.float32)
        )

        data = mjx.step(model_, data)
        pheromone.update(time_step)

        return robots, pheromone, data

    def __init__(
            self,
            settings: Settings,
            rngs: nnx.Rngs,
            data: mjx.Data,
            robots: BatchedRobots,
            pheromone: PheromoneField,
            pheromone_cell_ind: jax.Array,
            pheromone_cell_pos: jax.Array,
            controller: Controller
    ):
        self.rngs = rngs
        self.data = data
        self.robots = robots
        self.pheromone = pheromone
        self.controller = controller
        self.timer = 0

        self._jit_step = jax.jit(partial(
            SimulationData.__step,
            time_step=settings.Simulation.TIME_STEP,
            num_robots=robots.num_robots,
            pheromone_cell_ind=pheromone_cell_ind,
            pheromone_cell_pos=pheromone_cell_pos,
        ))

    def step(self, model: mjx.Model) -> 'SimulationData':
        input_ = jnp.zeros((self.robots.num_robots, 2), dtype=jnp.int32)
        input_ = input_.at[:, 0].set(
            jax.random.randint(self.rngs(), shape=(self.robots.num_robots,), minval=0, maxval=4, dtype=jnp.int32)
        )
        input_ = input_.at[:, 1].set(self.timer)

        controller_output = self.controller.forward(input_)

        updated_robots, updated_pheromone, updated_data = self._jit_step(
            model, self.data, self.pheromone, self.robots, controller_output
        )

        self.data = updated_data
        self.robots = updated_robots
        self.pheromone = updated_pheromone
        self.timer += 1

        return self

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
        mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_step(self.mj_model, mj_data)

        self.model = mjx.put_model(self.mj_model)
        data = mjx.put_data(self.mj_model, mj_data)

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
        pheromone_cell_ind = jnp.array(
            [(cell.index_x, cell.index_y) for cell in self.pheromone_cells], dtype=jnp.int32
        )
        pheromone_cell_pos = jnp.array(
            [(cell.pos[0], cell.pos[1]) for cell in self.pheromone_cells],
            dtype=jnp.float32
        )

        self.simulation_data = SimulationData(
            settings=settings,
            rngs=nnx.Rngs(0),
            data=data,
            robots=robots,
            pheromone=pheromone_field,
            pheromone_cell_ind=pheromone_cell_ind,
            pheromone_cell_pos=pheromone_cell_pos,
            controller=Controller(
                interval=int(1 / settings.Simulation.TIME_STEP),
                num_robots=settings.Robot.NUM
            )
        )

    def step(self):
        self.simulation_data = self.simulation_data.step(self.model)

    def render(self, img_buf: np.ndarray, pos: tuple[float, float, float], lookat: tuple[float, float, float]):
        if img_buf is None:
            return

        color_max = 1.0
        pheromone: jnp.ndarray = self.simulation_data.pheromone.values_gas
        for cell in self.pheromone_cells:
            pheromone_value = float(pheromone[cell.index_y, cell.index_x])
            rgb: tuple[float, float, float] = (pheromone_value / color_max, 0.0, 1 - pheromone_value / color_max)
            cell.set_color(*rgb, 0.5)

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
