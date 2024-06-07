import mujoco
import numpy as np

from lib.optimizer import MjcTaskInterface
from lib.sensor import DistanceMeasure

from .objects import Robot
from .brain import NeuralNetwork
from .hyper_parameters import HyperParameters, StaticParameters
from .xml import gen_xml


class _MuJoCoProcess:
    def __init__(self, bot_pos: tuple[float, float, float], brain: NeuralNetwork):
        self._measure = DistanceMeasure(
            HyperParameters.Robot.NUM_LIDAR, HyperParameters.Simulator.COLOR_MAP, StaticParameters.distance_measure_gain
        )

        xml = gen_xml(bot_pos)
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)

        self._bot = Robot(self.m, self.d, brain)
        self._goal_pos = np.array(HyperParameters.Environment.GOAL_POS)

        self._direction_buf = np.zeros((3, 1), dtype=np.float64)

    def get_bot(self):
        return self._bot

    def calc_step(self):
        mujoco.mj_step(self.m, self.d)

        self._bot.calc_direction(out=self._direction_buf)

        sight = self._measure.measure_with_img(
            self.m, self.d, self._bot.object_id, self._bot.body, self._direction_buf
        )
        brightness_img = np.dot(
            sight[0, :, :], np.array([[0.299], [0.587], [0.114]])
        ).reshape(
            (1, sight.shape[1])
        )

        self._bot.exec(self._direction_buf, sight, brightness_img)

        sub = self._bot.get_body().xpos[0:2] - self._goal_pos
        d = np.linalg.norm(sub)
        theta = np.dot(sub / d, self._direction_buf[0:2])[0]
        evaluation = d + HyperParameters.Evaluation.ANGLE_WEIGHT * 0.5 * abs(theta - 1)

        return evaluation


class Task(MjcTaskInterface):
    def __init__(self, bot_pos: list[tuple[float, float, float]], brain: NeuralNetwork):
        if len(bot_pos) <= 1:
            raise "Task needs a batch with a size lager than 2."

        self.bot_pos = bot_pos
        self.brain = brain
        self.mujoco = _MuJoCoProcess(self.bot_pos[0], self.brain)

    def get_model(self) -> mujoco.MjModel:
        return self.mujoco.m

    def get_data(self) -> mujoco.MjData:
        return self.mujoco.d

    def get_bot(self):
        return self.mujoco.get_bot()

    def calc_step(self) -> float:
        return self.mujoco.calc_step()

    def run(self) -> float:
        evaluations = np.zeros(HyperParameters.Simulator.TRY_COUNT)

        for _ in range(int(HyperParameters.Simulator.EPISODE / HyperParameters.Simulator.TIMESTEP + 0.5)):
            evaluations[0] += self.mujoco.calc_step()
        for i in range(1, len(self.bot_pos)):
            self.mujoco = _MuJoCoProcess(self.bot_pos[i], self.brain)
            for _ in range(int(HyperParameters.Simulator.EPISODE / HyperParameters.Simulator.TIMESTEP + 0.5)):
                evaluations[i] += self.mujoco.calc_step()

        return np.average(evaluations)

    def get_dummies(self):
        return []
