import abc

import numpy as np
import torch
import mujoco

from framework.prelude import *
from framework.backends import MujocoSTL


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

        self.parameters = parameters
        self.scores: list[Loss] = []
        self.sensors: list[list[SensorInterface]] = self.create_sensors()

        self.controller = controller
        self.input_ndarray = np.zeros((settings.Robot.NUM, 2 * 3), dtype=np.float32)
        self.input_tensor = torch.from_numpy(self.input_ndarray)

        torch.nn.utils.vector_to_parameters(
            torch.tensor(parameters, dtype=torch.float32),
            self.controller.parameters()
        )

        mujoco.mj_step(self.model, self.data)

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
        with torch.no_grad():
            input_ = self.create_input_for_controller()
            output = self.controller(input_)
            output_ndarray = output.numpy()

        for i, robot in enumerate(self.robot_values):
            robot.act(
                right_wheel=output_ndarray[i, 0],
                left_wheel=output_ndarray[i, 1]
            )

        mujoco.mj_step(self.model, self.data)

        loss = self.evaluation()
        self.scores.append(loss)

    def get_scores(self) -> list[float]:
        return [s.as_float() for s in self.scores]

    def calc_total_score(self) -> float:
        regularization_loss = self.settings.Loss.REGULARIZATION_COEFFICIENT * self.parameters.norm
        return sum(s.as_float() for s in self.scores) + regularization_loss
