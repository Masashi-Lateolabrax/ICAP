from collections.abc import Sequence

import mujoco

from mujoco_xml_generator.utils import DummyGeom

from lib.optimizer import MjcTaskInterface
from lib.utils import IntervalTimer

from .settings import Settings
from .core import Robot, RobotDebugBuf, lidar_c, trigonometric, World


class Task(MjcTaskInterface):
    def __init__(self, world: World, bot_pos: list[tuple[float, float, float]], para: Sequence[float]):
        self.world = world

        bot_pos_iter = iter(bot_pos)
        self.bots: list[Robot] = []
        para_offset = 0
        for factory, n in [
            (lidar_c.RobotFactory(True, True), Settings.Task.NUM_ROBOT_LI),
            (trigonometric.RobotFactory(True, True), Settings.Task.NUM_ROBOT_TR),
        ]:
            if n <= 0:
                continue
            dim = factory.get_dim()
            sub_para = para[para_offset:para_offset + dim]
            para_offset += dim
            for _, pos in zip(range(n), bot_pos_iter):
                bot_id = len(self.bots)
                bot = factory.manufacture(self.world, bot_id, sub_para)
                self.bots.append(bot)

        self.timer = IntervalTimer(Settings.Environment.Robot.THINK_INTERVAL)
        self._need_updating_panels = False

    def get_model(self) -> mujoco.MjModel:
        return self.world.model

    def get_data(self) -> mujoco.MjData:
        return self.world.data

    def enable_updating_panels(self, state_is_enable=True):
        self._need_updating_panels = state_is_enable

    def calc_step(self) -> float:
        mujoco.mj_step(self.world.model, self.world.data)

        for bot_id, bot in enumerate(self.bots):
            self.world.bot_pos[bot_id, :] = bot.get_body().xpos[0:2]

        if Settings.Environment.Pheromone.ENABLE_PHEROMONE:
            self.world.pheromone.update(
                Settings.Simulation.TIMESTEP,
                Settings.Simulation.PHEROMONE_ITER,
                self._need_updating_panels
            )

        exec_thinking: bool = self.timer.count(Settings.Simulation.TIMESTEP)
        for i, bot in enumerate(self.bots):
            bot.preparation(self.world)
            if exec_thinking:
                bot.update_state(self.world)
            bot.move(self.world)

        return Settings.Optimization.Evaluation.LOSS_FUNC(self.world.bot_pos, self.world.safezone_pos)

    def run(self) -> float:
        episode_length = int(Settings.Task.EPISODE_LENGTH / Settings.Simulation.TIMESTEP + 0.5)
        evaluation = 0.0
        for _ in range(episode_length):
            evaluation += self.calc_step()
        return evaluation

    def get_dummies(self) -> list[DummyGeom]:
        if not Settings.Environment.Pheromone.ENABLE_PHEROMONE:
            return []
        return self.world.pheromone.get_dummy_panels()

    def set_input_buf(self, bot_id: int, buf: RobotDebugBuf):
        for b in self.bots:
            b.set_input_buf(None)
        self.bots[bot_id].set_input_buf(buf)
