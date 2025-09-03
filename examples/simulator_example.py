import numpy as np
import mujoco
from mujoco import mjx

import jax
import jax.numpy as jnp
from flax import nnx
from flax.struct import dataclass as jax_dataclass

from framework.prelude import *
from framework.utils import GenericTkinterViewer
from framework.backends import SimulatorWithCtrl, ControllerInterface


class Controller(ControllerInterface):
    def __init__(self, parameter: int):  # parameter is the number of robots
        self.candidates = jnp.array([
            [1., -1.],
            [-1., 1.],
            [1., 1.],
            [-0.5, -0.5],
        ])
        self.rngs = nnx.Rngs(0)
        self.state = jnp.zeros((parameter, 3), dtype=jnp.float32)

    def __call__(self, x: jax.Array) -> jax.Array:
        do_update = jax.random.randint(self.rngs(), (1,), minval=0, maxval=100)
        select = jax.random.randint(self.rngs(), (x.shape[0],), minval=0, maxval=4)

        x = jax.lax.cond(
            do_update[0] < 1,
            lambda _: self.candidates[select, :2],
            lambda _: self.state[:, :2],
            operand=None
        )
        self.state = self.state.at[:, :2].set(x)
        self.state = self.state.at[:, 2].add(1)

        return self.state

    @staticmethod
    def dim():
        return 0


@jax_dataclass
class Simulator:
    _sim: SimulatorWithCtrl

    @property
    def data(self) -> mjx.Data:
        return self._sim.data

    @property
    def food_items(self) -> BatchedFood:
        return self._sim.food_items

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
            "controller": controller
        }
        sim = self._sim.update(**parent_kwargs)

        this_kwargs = {
            "_sim": sim,
        }
        kwargs = {k: v for k, v in this_kwargs.items() if v is not None}
        if not kwargs:
            return self

        return self.replace(**kwargs)

    @classmethod
    def new(cls, settings: Settings, controller: Controller, rngs: jax.Array) -> tuple[mujoco.MjModel, 'Simulator']:
        mj_model, sim = SimulatorWithCtrl.new(settings, controller, rngs)
        return mj_model, cls(_sim=sim)

    def add_pheromone(self, positions: jax.Array, amounts: jax.Array) -> 'Simulator':
        new_sim = self._sim.add_pheromone(positions, amounts)
        return self.replace(_sim=new_sim)

    @staticmethod
    @nnx.jit
    def _step(this: 'Simulator') -> 'Simulator':
        this: "Simulator" = this.replace(_sim=this._sim.step())

        nest_dir = -this.food_items.positions
        nest_dir = nest_dir.at[:, 2].set(0.0)
        force = nest_dir / (jnp.linalg.norm(nest_dir, axis=1, keepdims=True) + 1e-6) * 50.0
        new_data = this.food_items.set_force(this.data, jnp.arange(force.shape[0]), force)

        return this.update(data=new_data)

    def step(self) -> 'Simulator':
        return Simulator._step(self)

    @staticmethod
    @nnx.jit
    def _step_n(this: "Simulator", n: int) -> "Simulator":
        def body_fn(_i, sim: "Simulator"):
            return Simulator._step(sim)

        this = jax.lax.fori_loop(0, n, body_fn, this)
        return this

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
        self._sim.render(mj_model, img_buf, pos, lookat, max_geom, max_pheromone)

    def reset(self, controller: ControllerInterface = None, rngs: jax.Array = None) -> 'Simulator':
        new_sim = self._sim.reset(rngs)
        this = self.replace(_sim=new_sim)
        return this.update(
            controller=controller
        )


def jaxable_example():
    cpu_device = jax.devices("cpu")[0]

    settings = Settings()

    settings.Render.RENDER_WIDTH = 480
    settings.Render.RENDER_HEIGHT = 320

    settings.Robot.NUM = 1
    settings.Food.NUM = 1

    rngs = jax.random.PRNGKey(0)
    mj_model, backend = Simulator.new(
        settings,
        Controller(settings.Robot.NUM),
        rngs
    )
    backend = jax.device_put(backend, cpu_device)

    viewer = GenericTkinterViewer(mj_model, settings, backend)
    viewer.run()


if __name__ == '__main__':
    jaxable_example()
