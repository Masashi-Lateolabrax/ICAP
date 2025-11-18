import numpy as np
import torch
import mujoco

from framework.prelude import *
from framework.environment import rand_food_pos
from framework.backends import BasicSimulator
from framework.utils import Timer
from framework.sensor import DirectionSensor, DepthSensor, VelocitySensor

from .controller import Controller
from .loss import Loss


class Simulator(BasicSimulator):
    @staticmethod
    def _create_sensors(
            settings: Settings,
            robot_values: RobotValues,
            nest_site: mujoco._specs.MjsSite,
            model: mujoco.MjModel,
            data: mujoco.MjData
    ) -> list[SensorInterface]:
        return [
            DirectionSensor(
                robot=robot_values,
                target_site=nest_site,
                target_radius=settings.Nest.RADIUS
            ),
            VelocitySensor(
                robot=robot_values
            ),
            DepthSensor(
                robot=robot_values,
                model=model,
                data=data,
                num_rays=settings.Robot.DEPTH_SENSOR_NUM_RAYS,
                max_range=settings.Robot.DEPTH_SENSOR_MAX_RANGE,
                offset=settings.Robot.RADIUS
            )
        ]

    def __init__(self, settings: Settings, individual: Individual, render: bool = False):
        super().__init__(settings, render)

        self.settings = settings

        self.controller = Controller(settings, individual)

        self.timer = Timer(settings.Robot.THINK_INTERVAL / settings.Simulation.TIME_STEP)
        self.rng = np.random.default_rng(individual.generation)

        self.parameters: Individual = individual
        self.scores: list[Loss] = []
        self._max_pheromone: float = 0.0

        self.sensors: list[list[SensorInterface]] = [
            self._create_sensors(settings, r, self.nest_site, self.model, self.data)
            for r in self.robot_values
        ]

        self.dummy_foods: list[DummyFoodValues] = []

        # Input dimensions: direction(2) + velocity(3) + depth(N) + pheromone(3)
        # where N = DEPTH_SENSOR_NUM_RAYS
        input_dim = 2 + 3 + settings.Robot.DEPTH_SENSOR_NUM_RAYS + 3
        self.input_ndarray = np.zeros((settings.Robot.NUM, input_dim), dtype=np.float32)
        self.output_ndarray = np.zeros((settings.Robot.NUM, 3), dtype=np.float32)
        self.input_tensor = torch.from_numpy(self.input_ndarray)

        # Pre-allocated arrays for step() method optimization
        self._robot_positions = np.zeros((settings.Robot.NUM, 2), dtype=np.float32)
        self._robot_v_direction = np.zeros((settings.Robot.NUM, 2), dtype=np.float32)
        self._robot_h_direction = np.zeros((settings.Robot.NUM, 2), dtype=np.float32)

        mujoco.mj_step(self.model, self.data)

    def get_max_gas_pheromone(self) -> float:
        return self._max_pheromone

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
        num_rays = self.settings.Robot.DEPTH_SENSOR_NUM_RAYS
        for i, sensors in enumerate(self.sensors):
            self.input_ndarray[i, 0:2] = sensors[0].get()  # direction
            self.input_ndarray[i, 2:5] = sensors[1].get()  # velocity
            self.input_ndarray[i, 5:5 + num_rays] = sensors[2].get()  # depth sensor

        # Pheromone data (if available)
        if self._pheromone_field is not None:
            pheromone_idx = 5 + num_rays
            self.input_ndarray[:, pheromone_idx] = self.get_pheromone(self._robot_positions) / 3.5

            pheromone_grad = self.get_pheromone_grad(self._robot_positions)
            self.input_ndarray[:, pheromone_idx + 1] = np.sum(pheromone_grad * self._robot_v_direction, axis=1)
            self.input_ndarray[:, pheromone_idx + 2] = np.sum(pheromone_grad * self._robot_h_direction, axis=1)

        return self.input_tensor

    def step(self):
        # Use pre-allocated arrays to avoid memory allocation overhead
        for i, robot in enumerate(self.robot_values):
            self._robot_positions[i] = robot.xpos
            self._robot_v_direction[i] = robot.xdirection
            self._robot_h_direction[i, 0] = robot.xdirection[1]
            self._robot_h_direction[i, 1] = -robot.xdirection[0]

        if self.timer.tick():
            with torch.no_grad():
                input_ = self.create_input_for_controller()
                output = self.controller.forward(input_)
                self.output_ndarray = output.numpy()

        for i, robot in enumerate(self.robot_values):
            robot.act(
                right_wheel=self.output_ndarray[i, 0],
                left_wheel=self.output_ndarray[i, 1]
            )

        if self._pheromone_field is not None:
            self.add_pheromone(self._robot_positions, self.output_ndarray[:, 2] * self.settings.Robot.MAX_PHEROMONE_SECRETION)
            self._pheromone_field.step()

            max_pheromone = self._pheromone_field.get_max_value()
            self._max_pheromone = max(self._max_pheromone, max_pheromone)

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
