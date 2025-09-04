from typing import Self

import numpy as np
import mujoco
from mujoco import mjx

import jax
import jax.numpy as jnp
from flax import nnx
from flax.struct import dataclass as jax_dataclass

from framework.prelude import *
from framework.utils import GenericTkinterViewer
from framework.backends import SimulatorWithCtrl, ControllerInterface, RobotOutputs


class Controller(ControllerInterface):
    def __init__(self, parameter: int):  # parameter is the number of robots
        self.candidates = jnp.array([
            [1., -1.],
            [-1., 1.],
            [1., 1.],
            [-0.5, -0.5],
        ])
        self.rngs = nnx.Rngs(0)
        self.state = jnp.ones((parameter, 3), dtype=jnp.float32)

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

        return self.state

    def forward(self, x: RobotInputs) -> RobotOutputs:
        x = x.as_matrix()
        x = self.__call__(x)
        return RobotOutputs(
            left_wheel=x[:, 0],
            right_wheel=x[:, 1],
            pheromone=x[:, 2],
        )

    def reset(self) -> Self:
        return self

    @staticmethod
    def dim():
        return 0


@jax_dataclass
class Simulator(SimRenderTrait):
    _parent_sim: SimulatorWithCtrl

    @property
    def data(self) -> mjx.Data:
        return self._parent_sim.data

    @property
    def food_items(self) -> BatchedFood:
        return self._parent_sim.food_items

    def _update_parent(self, **kwargs: dict) -> Self:
        return self.replace(
            _parent_sim=self._parent_sim.update(**kwargs)
        )

    def update(
            self,
            data: mjx.Data = None,
            food_items: BatchedFood = None,
            controller: Controller = None,
            **kwargs
    ) -> Self:
        kwargs["data"] = data
        kwargs["food_items"] = food_items
        kwargs["controller"] = controller
        return self._update(**kwargs)

    @classmethod
    def new(cls, settings: Settings, controller: Controller, rngs: jax.Array) -> tuple[mujoco.MjModel, 'Simulator']:
        mj_model, parent_sim = SimulatorWithCtrl.new(settings, controller, rngs)
        return mj_model, cls(_parent_sim=parent_sim)

    @staticmethod
    @nnx.jit
    def _step(this: 'Simulator') -> 'Simulator':
        this = this.update(
            _parent_sim=this._parent_sim.step()
        )

        nest_dir = -this.food_items.positions
        nest_dir = nest_dir.at[:, 2].set(0.0)
        force = nest_dir / (jnp.linalg.norm(nest_dir, axis=1, keepdims=True) + 1e-6) * 50.0
        new_data = this.food_items.set_force(this.data, jnp.arange(force.shape[0]), force)

        return this.update(data=new_data)

    def step(self) -> Self:
        return Simulator._step(self)

    @staticmethod
    @nnx.jit
    def _step_n(this: "Simulator", n: int) -> "Simulator":
        def body_fn(_i, sim: "Simulator"):
            return Simulator._step(sim)

        this = jax.lax.fori_loop(0, n, body_fn, this)
        return this

    def step_n(self, n: int) -> Self:
        return Simulator._step_n(self, n)

    def reset(self) -> Self:
        parent_sim = self._parent_sim.reset()
        return self.update(_parent_sim=parent_sim)

    def render(self, img_buf: np.ndarray, camera: mujoco.MjvCamera, renderer: mujoco.Renderer):
        self._parent_sim.render(img_buf, camera, renderer)


def jaxable_example():
    cpu_device = jax.devices("cpu")[0]

    settings = Settings()

    settings.Render.RENDER_WIDTH = 480
    settings.Render.RENDER_HEIGHT = 320

    settings.Robot.NUM = 1
    settings.Robot.INITIAL_POSITION = []

    settings.Food.NUM = 1
    settings.Food.INITIAL_POSITION = []

    settings.Pheromone.CELL_SIZE = 0.5
    settings.Pheromone.WIDTH_NUM = int(settings.Simulation.WORLD_WIDTH / settings.Pheromone.CELL_SIZE)
    settings.Pheromone.HEIGHT_NUM = int(settings.Simulation.WORLD_HEIGHT / settings.Pheromone.CELL_SIZE)

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
