import numpy as np
import torch
import mujoco

from framework.prelude import *
from framework.environment import rand_food_pos
from framework.backends import BasicSimulator
from framework.utils import Timer
from framework.sensor import PreprocessedOmniSensor, DirectionSensor

from .controller import Controller
from .loss import Loss


class Simulator(BasicSimulator):
    @staticmethod
    def _create_sensors(
            settings: Settings,
            robot_values: RobotValues,
            all_robot_values: list[RobotValues],
            all_food_values: list[FoodValues],
            nest_site: mujoco._specs.MjsSite
    ) -> list[SensorInterface]:
        return [
            PreprocessedOmniSensor(
                robot=robot_values,
                d_gain=settings.Robot.ROBOT_SENSOR_GAIN,
                offset=settings.Robot.RADIUS * 2,
                target_sites=[other.site for j, other in enumerate(all_robot_values) if other is robot_values]
            ),
            PreprocessedOmniSensor(
                robot=robot_values,
                d_gain=settings.Robot.FOOD_SENSOR_GAIN,
                offset=settings.Robot.RADIUS + settings.Food.RADIUS,
                target_sites=[food.site for food in all_food_values]
            ),
            DirectionSensor(
                robot=robot_values,
                target_site=nest_site,
                target_radius=settings.Nest.RADIUS
            )
        ]

    def __init__(self, settings: Settings, individual: Individual, render: bool = False):
        super().__init__(settings, render)

        self.settings = settings

        self.controller = Controller(individual)

        self.timer = Timer(settings.Robot.THINK_INTERVAL / settings.Simulation.TIME_STEP)
        self.rng = np.random.default_rng(individual.generation)

        self.parameters: Individual = individual
        self.scores: list[Loss] = []

        self.sensors: list[list[SensorInterface]] = [
            self._create_sensors(settings, r, self.robot_values, self.food_values, self.nest_site)
            for r in self.robot_values
        ]

        self.dummy_foods: list[DummyFoodValues] = []

        self.input_ndarray = np.zeros((settings.Robot.NUM, 2 * 3 + 1), dtype=np.float32)
        self.output_ndarray = np.zeros((settings.Robot.NUM, 3), dtype=np.float32)
        self.input_tensor = torch.from_numpy(self.input_ndarray)

        mujoco.mj_step(self.model, self.data)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.dummy_foods.clear()

    def _respawn_food(self, food: FoodValues):
        dummy_food = DummyFoodValues(food)
        self.dummy_foods.append(dummy_food)

        invalid_area = [
            (Position(self.nest_site.xpos[0], self.nest_site.xpos[1]), self.settings.Nest.RADIUS)
        ]

        for f in self.food_values:
            invalid_area.append(
                (f.position, self.settings.Food.RADIUS)
            )

        new_position = rand_food_pos(self.settings, invalid_area, self.rng)

        food_joint = food.joint

        food_joint.qpos[0] = new_position.x
        food_joint.qpos[1] = new_position.y
        food_joint.qpos[2] = 1
        food_joint.qvel[:] = 0.0
        food_joint.qacc[:] = 0.0

    def create_input_for_controller(self):
        for i, sensors in enumerate(self.sensors):
            self.input_ndarray[i, 0:2] = sensors[0].get()
            self.input_ndarray[i, 2:4] = sensors[1].get()
            self.input_ndarray[i, 4:6] = sensors[2].get()

        return self.input_tensor

    def step(self):
        robot_positions = np.array([robot.xpos for robot in self.robot_values])

        if self.timer.tick():
            with torch.no_grad():
                input_ = self.create_input_for_controller()
                if self._pheromone_field is not None:
                    self.input_ndarray[:, 6] = self.get_pheromone(robot_positions)

                output = self.controller.forward(input_)
                self.output_ndarray = output.numpy()

        for i, robot in enumerate(self.robot_values):
            robot.act(
                right_wheel=self.output_ndarray[i, 0],
                left_wheel=self.output_ndarray[i, 1]
            )

        if self._pheromone_field is not None:
            self.add_pheromone(robot_positions, self.output_ndarray[:, 2])
            self._pheromone_field.add_liquid_by_cell(self._pheromone_cells)
            self._pheromone_field.update(self.settings.Simulation.TIME_STEP)

        for food in self.food_values:
            if np.linalg.norm(food.xpos - self.nest_site.xpos[0:2]) <= self.settings.Nest.RADIUS:
                self._respawn_food(food)

        mujoco.mj_step(self.model, self.data)

        robot_positions = [r.xpos for r in self.robot_values]
        food_positions = [f.xpos for f in self.food_values] + [f.xpos for f in self.dummy_foods]
        nest_position = self.nest_site.xpos
        self.scores.append(
            Loss(
                self.settings,
                robot_positions=robot_positions,
                food_positions=food_positions,
                nest_position=nest_position
            )
        )

    def get_scores(self) -> list[float]:
        return [s.as_float() for s in self.scores]

    def calc_total_score(self) -> float:
        regularization_loss = self.settings.Loss.REGULARIZATION_COEFFICIENT * self.parameters.norm
        return sum(s.as_float() for s in self.scores) + regularization_loss
