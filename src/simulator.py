import numpy as np
import torch
import mujoco

from framework.prelude import *
from framework.backends import MujocoBackend
from framework.sensor import PreprocessedOmniSensor, DirectionSensor

from loss import Loss
from controller import RobotNeuralNetwork


class Simulator(MujocoBackend):
    def __init__(self, settings: Settings, parameters: Individual, render: bool = False):
        super().__init__(settings, render)

        self.parameters = parameters
        self.scores: list[Loss] = []
        self.sensors: list[tuple[SensorInterface]] = self._create_sensors()

        self.controller = RobotNeuralNetwork(parameters)
        self.input_ndarray = np.zeros((settings.Robot.NUM, 2 * 2 + 1), dtype=np.float32)
        self.input_tensor = torch.from_numpy(self.input_ndarray)

        mujoco.mj_step(self.model, self.data)

    def _create_sensors(self) -> list[tuple[SensorInterface]]:
        sensors = []
        for i, robot in enumerate(self.robot_values):
            sensor_tuple = (
                PreprocessedOmniSensor(
                    robot,
                    self.settings.Robot.ROBOT_SENSOR_GAIN,
                    self.settings.Robot.RADIUS * 2,
                    [other.site for j, other in enumerate(self.robot_values) if j != i]
                ),
                PreprocessedOmniSensor(
                    robot,
                    self.settings.Robot.FOOD_SENSOR_GAIN,
                    self.settings.Robot.RADIUS + self.settings.Food.RADIUS,
                    [food.site for food in self.food_values]
                ),
                DirectionSensor(
                    robot, self.nest_site, self.settings.Nest.RADIUS
                )
            )
            sensors.append(sensor_tuple)
        return sensors

    def _create_input_for_controller(self):
        for i, sensors in enumerate(self.sensors):
            self.input_ndarray[i, 0:2] = sensors[0].get()
            self.input_ndarray[i, 2:4] = sensors[1].get()
            self.input_ndarray[i, 4] = sensors[2].get()
        return self.input_tensor

    def evaluation(self) -> Loss:
        robot_positions = [r.xpos for r in self.robot_values]
        food_positions = [f.xpos for f in self.food_values]
        nest_position = self.nest_site.xpos
        return Loss(
            self.settings,
            robot_positions=robot_positions,
            food_positions=food_positions,
            nest_position=nest_position
        )

    def step(self):
        with torch.no_grad():
            input_ = self._create_input_for_controller()
            output = self.controller.forward(input_)
            output_ndarray = output.numpy()

        for i, robot in enumerate(self.robot_values):
            robot.act(
                right_wheel=output_ndarray[i, 0],
                left_wheel=output_ndarray[i, 1]
            )

        mujoco.mj_step(self.model, self.data)

        loss = self.evaluation()
        self.scores.append(loss)

    def scores(self) -> list[float]:
        return [s.as_float() for s in self.scores]

    def total_score(self) -> float:
        regularization_loss = self.settings.Optimization.REGULARIZATIONl_LOSS * self.parameters.norm
        return sum(s.as_float() for s in self.scores) + regularization_loss
