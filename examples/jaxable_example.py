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

@jax_dataclass
class Simulator:
    _env_sim: BasicSimulatorWithEnv

    controller: Controller

    @property
    def data(self) -> mjx.Data:
        return self._env_sim.data

    @property
    def robots(self) -> BatchedRobots:
        return self._env_sim.robots

    @property
    def robot_inputs(self) -> jax.Array:
        return self._env_sim.robot_inputs

    @property
    def loss(self) -> jax.Array:
        return self._env_sim.loss

    def update(
            self,
            data: mjx.Data = None,
            robots: BatchedRobots = None,
            robot_inputs: jax.Array = None,
            loss: jax.Array = None,

            controller: Controller = None
    ) -> 'Simulator':
        parent_kwargs = {
            "data": data,
            "robots": robots,
            "robot_inputs": robot_inputs,
            "loss": loss,
        }
        env_sim = self._env_sim.update(**parent_kwargs)

        this_kwargs = {
            "_env_sim": env_sim,
            "controller": controller,
        }
        kwargs = {k: v for k, v in this_kwargs.items() if v is not None}
        if not kwargs:
            return self

        return self.replace(**kwargs)

    @classmethod
    def new(cls, settings: Settings, rngs: jax.Array) -> 'Simulator':
        sim = BasicSimulatorWithEnv.new(settings, rngs)
        controller = Controller(settings.Robot.NUM)
        return cls(
            _env_sim=sim,
            controller=controller,
        )

    def add_pheromone(self, positions: jax.Array, amounts: jax.Array) -> 'Simulator':
        new_env_sim = self._env_sim.add_pheromone(positions, amounts)
        return self.replace(_env_sim=new_env_sim)

    @staticmethod
    @nnx.jit
    def _step(this: 'Simulator') -> 'Simulator':
        this: "Simulator" = this.replace(_env_sim=this._env_sim.step())

        output = this.controller(this.robot_inputs)
        new_data = this.robots.set_ctrl(this.data, output)

        this = this.add_pheromone(
            this.robots.positions,
            jnp.ones((this.robots.num_robots,), dtype=jnp.float32)
        )

        return this.update(data=new_data)

    def step(self) -> 'Simulator':
        return Simulator._step(self)

    @staticmethod
    @nnx.jit
    def _step_n(simulator: "Simulator", n: int) -> "Simulator":
        def body_fn(_i, sim: "Simulator"):
            return Simulator._step(sim)

        new_simulator = jax.lax.fori_loop(0, n, body_fn, simulator)
        return new_simulator

    def step_n(self, n: int) -> 'Simulator':
        return Simulator._step_n(self, n)

    def render(
            self,
            mj_model: mujoco.MjModel,
            img_buf: np.ndarray,
            pos: tuple[float, float, float],
            lookat: tuple[float, float, float],
            max_geom=100,
            max_pheromone=1.0
    ):
        self._env_sim.render(mj_model, img_buf, pos, lookat, max_geom, max_pheromone)

    def reset(self, individual: jax.Array = None, rngs: jax.Array = None) -> 'Simulator':
        new_env_sim = self._env_sim.reset(rngs)
        this = self.replace(_env_sim=new_env_sim)

        controller = Controller(individual) if individual is not None else None
        return this.update(
            individual=individual,
            controller=controller
        )


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
