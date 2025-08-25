import mujoco
import jax
import jax.numpy as jnp

from framework.prelude import *
from framework.utils import GenericTkinterViewer
from framework.backends import BasicMuJoCoSimulatorWithEnv
from framework.utils import Timer


class Controller:
    def __init__(self, interval: int, num_robots: int):
        self.timer = Timer(interval)
        self.output = jnp.zeros((num_robots, 2))
        self.rng_key = jax.random.PRNGKey(42)

        action_map = jnp.array([
            [-1.0, 1.0],  # LEFT
            [1.0, -1.0],  # RIGHT
            [1.0, 1.0],  # FORWARD
            [-0.8, -0.8]  # BACKWARD
        ])

        def gen_action(key):
            actions = jax.random.randint(key, (num_robots,), 0, 4)
            return action_map[actions]

        self._gen_action = jax.jit(gen_action)

    def forward(self, _input):
        if self.timer.tick():
            self.rng_key, subkey = jax.random.split(self.rng_key)
            self.output = self._gen_action(subkey)
        return self.output


class Simulator(BasicMuJoCoSimulatorWithEnv):
    def __init__(self, settings: Settings, render: bool = False):
        super().__init__(settings, render)
        self.controller = Controller(int(1 / settings.Simulation.TIME_STEP), settings.Robot.NUM)
        mujoco.mj_step(self.model, self.data)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

    def get_scores(self) -> list[float]:
        return []

    def calc_total_score(self) -> float:
        return 0.0

    def step(self):
        self.robots.update(self.data)

        outputs = self.controller.forward(None)
        self.data = self.robots.set_ctrl(self.data, outputs)
        mujoco.mj_step(self.model, self.data)


def mujoco_example():
    settings = Settings()
    settings.Render.RENDER_WIDTH = 480
    settings.Render.RENDER_HEIGHT = 320

    settings.Robot.NUM = 3
    settings.Food.NUM = 1

    viewer = GenericTkinterViewer(
        settings,
        Simulator(settings, render=True),
    )
    viewer.run()


if __name__ == '__main__':
    mujoco_example()
