import math
from typing import Callable

from jax import numpy as jnp
from flax import nnx

from framework.prelude import Settings, SimulatorBackend, Individual, SensorInterface, DummyFoodValues
from framework.sensor import DirectionSensor, PreprocessedOmniSensor
from framework.backends import MujocoStlGpu
from framework.utils import Timer

from .loss import Loss
from .controller import RobotNeuralNetwork


class Simulator(MujocoStlGpu):
    def __init__(self, settings: Settings, individual: Individual, render: bool):
        super().__init__(settings, render)

        rngs = nnx.Rngs(0)
        self.controller = RobotNeuralNetwork(individual, rngs)

        self.timer = Timer(settings.Robot.THINK_INTERVAL / settings.Simulation.TIME_STEP)
        self.rng = nnx.Rngs(individual.generation)

        self.scores: list[Loss] = []
        self.sensors: list[list[SensorInterface]] = self.create_sensors()

        self.dummy_foods: list[DummyFoodValues] = []

        super().step()

    def create_sensors(self) -> list[list[SensorInterface]]:
        sensors = []
        for i, robot in enumerate(self.robot_values):
            sensor_tuple = [
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
            ]
            sensors.append(sensor_tuple)
        return sensors

    def create_input_for_controller(self):
        input_data = []
        for i, sensors in enumerate(self.sensors):
            robot_input = jnp.concatenate([
                sensors[0].get(),
                sensors[1].get(),
                sensors[2].get()
            ])
            input_data.append(robot_input)

        return jnp.stack(input_data)

    def evaluation(self) -> Loss:
        robot_positions = jnp.stack([r.xpos for r in self.robot_values], axis=0)
        food_positions = jnp.stack([f.xpos for f in self.food_values] + [f.xpos for f in self.dummy_foods], axis=0)
        nest_position = self.nest_site.xpos
        return Loss(
            robot_positions=robot_positions,
            food_positions=food_positions,
            nest_position=nest_position
        )


class SimulatorBuilder:
    def __init__(self, settings):
        self.settings = settings

    def build(self, individual: Individual):
        return Simulator(self.settings, individual, False)


class EvaluationFunction:
    def __init__(self, settings: Settings, simulator_builder: Callable[[Individual], SimulatorBackend]):
        self.settings = settings
        self.simulator_builder = simulator_builder

    def run(self, individual: Individual):
        backend = self.simulator_builder(individual)
        for _ in range(math.ceil(self.settings.Simulation.TIME_LENGTH / self.settings.Simulation.TIME_STEP)):
            backend.step()
        return backend.calc_total_score()
