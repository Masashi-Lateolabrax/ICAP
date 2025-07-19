import abc

import numpy as np
import torch
import mujoco

from framework.prelude import *
from framework.environment import rand_food_pos
from framework.backends import MujocoSTL
from framework.utils import Timer


class Loss(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, settings: Settings, robot_positions, food_positions, nest_position):
        raise NotImplementedError("Subclasses should implement this method.")

    @abc.abstractmethod
    def as_float(self) -> float:
        raise NotImplementedError("Subclasses should implement this method.")


class Simulator(MujocoSTL, abc.ABC):
    def __init__(self, settings: Settings, parameters: Individual, controller: torch.nn.Module, render: bool = False):
        super().__init__(settings, render)

        self.timer = Timer(settings.Robot.THINK_INTERVAL / settings.Simulation.TIME_STEP)
        self.rng = np.random.default_rng(parameters.generation)

        self.parameters = parameters
        self.scores: list[Loss] = []
        self.sensors: list[list[SensorInterface]] = self.create_sensors()

        self.dummy_foods: list[DummyFoodValues] = []

        self.controller = controller
        self.input_ndarray = np.zeros((settings.Robot.NUM, 2 * 3), dtype=np.float32)
        self.output_ndarray = np.zeros((settings.Robot.NUM, 2), dtype=np.float32)
        self.input_tensor = torch.from_numpy(self.input_ndarray)

        torch.nn.utils.vector_to_parameters(
            torch.tensor(parameters.as_ndarray, dtype=torch.float32),
            self.controller.parameters()
        )

        mujoco.mj_step(self.model, self.data)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.dummy_foods.clear()

    @abc.abstractmethod
    def create_sensors(self) -> list[list[SensorInterface]]:
        raise NotImplementedError("Subclasses should implement this method.")

    @abc.abstractmethod
    def create_input_for_controller(self):
        raise NotImplementedError("Subclasses should implement this method.")

    @abc.abstractmethod
    def evaluation(self) -> Loss:
        raise NotImplementedError("Subclasses should implement this method.")

    def step(self):
        if not self.timer.tick():
            with torch.no_grad():
                input_ = self.create_input_for_controller()
                output = self.controller(input_)
                self.output_ndarray = output.numpy()

        for i, robot in enumerate(self.robot_values):
            robot.act(
                right_wheel=self.output_ndarray[i, 0],
                left_wheel=self.output_ndarray[i, 1]
            )

        self.check_and_respawn_food()

        mujoco.mj_step(self.model, self.data)

        loss = self.evaluation()
        self.scores.append(loss)

    def get_scores(self) -> list[float]:
        return [s.as_float() for s in self.scores]

    def calc_total_score(self) -> float:
        regularization_loss = self.settings.Loss.REGULARIZATION_COEFFICIENT * np.linalg.norm(self.parameters.as_ndarray)
        return sum(s.as_float() for s in self.scores) + regularization_loss

    def _is_food_in_nest(self, food_values: FoodValues) -> bool:
        food_pos = food_values.xpos
        nest_pos = self.nest_site.xpos[0:2]
        nest_radius = self.settings.Nest.RADIUS

        distance = np.linalg.norm(food_pos - nest_pos)
        return distance <= nest_radius

    def _respawn_food(self, food_values: FoodValues):
        dummy_food = DummyFoodValues(food_values)
        self.dummy_foods.append(dummy_food)

        invalid_area = [
            (Position(self.nest_site.xpos[0], self.nest_site.xpos[1]), self.settings.Nest.RADIUS)
        ]

        for food in self.food_values:
            if food is not food_values:
                invalid_area.append(
                    (food.position, self.settings.Food.RADIUS)
                )

        new_position = rand_food_pos(self.settings, invalid_area, self.rng)

        food_joint = food_values.joint

        food_joint.qpos[0] = new_position.x
        food_joint.qpos[1] = new_position.y
        food_joint.qpos[2] = 1
        food_joint.qvel[:] = 0.0
        food_joint.qacc[:] = 0.0

    def check_and_respawn_food(self):
        for food in self.food_values:
            if self._is_food_in_nest(food):
                self._respawn_food(food)
