import mujoco
import numpy as np
import torch

from lib.optimizer import TaskInterface
from lib.sensor import trigono_omni_sensor, direction_sensor

from .objects import Robot, Food
from .brain import NeuralNetwork
from .hyper_parameters import HyperParameters
from .xml import gen_xml


class _PositionManager:
    def __init__(self, num_bot, num_food):
        self.bot_pos = np.zeros((num_bot, 2))
        self.food_pos = np.zeros((num_food, 2))
        self.nest_pos = np.array(HyperParameters.Environment.NEST_POS)
        self.score = 0

    def get_bot_pos(self, exception: int | None = None):
        if exception is None:
            return self.bot_pos
        i = [i != exception for i in range(self.bot_pos.shape[0])]
        return self.bot_pos[i, :]

    def get_food_pos(self, exception: int | None = None):
        if exception is None:
            return self.food_pos
        i = [i != exception for i in range(self.food_pos.shape[0])]
        return self.food_pos[i, :]

    def update(self, bots: list[Robot], food: list[Food]):
        for i, b in enumerate(bots):
            p = b.get_body().xpos[0:2]
            self.bot_pos[i, :] = p

        for i, f in enumerate(food):
            p = f.get_geom().xpos[0:2]
            self.food_pos[i, :] = p

    def evaluate(self):
        dif_food_nest_score = np.sum(
            np.linalg.norm(self.food_pos - self.nest_pos, axis=1)
        )

        dif_food_robot_score = 0
        for f in self.food_pos:
            d = np.sum((self.bot_pos - f) ** 2, axis=1)
            dif_food_robot_score -= np.sum(
                np.exp(-d / HyperParameters.Evaluation.FOOD_RANGE)
            )

        dif_food_nest_score *= HyperParameters.Evaluation.FOOD_NEST_GAIN / len(self.food_pos)
        dif_food_robot_score *= HyperParameters.Evaluation.FOOD_ROBOT_GAIN / (len(self.food_pos) * len(self.bot_pos))

        return dif_food_robot_score + dif_food_nest_score


class _MuJoCoProcess:
    def __init__(self, brain: NeuralNetwork):
        xml = gen_xml()
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)

        self._bots = [
            Robot(self.m, self.d, i, brain) for i in range(len(HyperParameters.Environment.BOT_POS))
        ]
        self._food = [
            Food(self.m, self.d, i) for i in range(len(HyperParameters.Environment.FOOD_POS))
        ]

        self._positions = _PositionManager(len(self._bots), len(self._food))

        self.input_buf = torch.zeros((6, 1))
        self._direction_buf = np.zeros((3, 1), dtype=np.float64)

        self.last_exec_robot_index = -1

    def get_bots(self):
        return self._bots

    def calc_step(self):
        mujoco.mj_step(self.m, self.d)

        self._positions.update(self._bots, self._food)

        bot_list = list(enumerate(self._bots))
        bot_list[self.last_exec_robot_index], bot_list[-1] = bot_list[-1], bot_list[self.last_exec_robot_index]
        for i, bot in bot_list:
            bot.calc_direction(out=self._direction_buf)
            self.input_buf[0], self.input_buf[1] = trigono_omni_sensor(
                bot.get_body().xpos[0:2], self._direction_buf[0:2, 0], self._positions.get_bot_pos(i),
                lambda d: 1 / (d * HyperParameters.Robot.SENSOR_PRECISION + 1)
            )
            self.input_buf[2], self.input_buf[3] = trigono_omni_sensor(
                bot.get_body().xpos[0:2], self._direction_buf[0:2, 0], self._positions.food_pos,
                lambda d: 1 / (d * HyperParameters.Robot.SENSOR_PRECISION + 1)
            )
            self.input_buf[4], self.input_buf[5] = direction_sensor(
                bot.get_body().xpos[0:2], self._direction_buf[0:2, 0],
                self._positions.nest_pos, HyperParameters.Environment.NEST_SIZE
            )

            bot.exec(self._direction_buf, self.input_buf)

        return self._positions.evaluate()


class Task(TaskInterface):
    def __init__(self, brain: NeuralNetwork, ):
        self.mujoco = _MuJoCoProcess(brain)

    def get_model(self) -> mujoco.MjModel:
        return self.mujoco.m

    def get_data(self) -> mujoco.MjData:
        return self.mujoco.d

    def get_bots(self):
        return self.mujoco.get_bots()

    def calc_step(self) -> float:
        return self.mujoco.calc_step()

    def run(self) -> float:
        evaluations = 0
        for _ in range(int(HyperParameters.Simulator.EPISODE / HyperParameters.Simulator.TIMESTEP + 0.5)):
            evaluations += self.calc_step()
        return np.average(evaluations)
