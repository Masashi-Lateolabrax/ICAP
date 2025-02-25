import mujoco
import numpy as np

from libs import optimizer

from scheme.pushing_food_with_pheromone.lib.world import World, WorldBuilder1
from scheme.pushing_food_with_pheromone.lib.objects.robot import Robot
from scheme.pushing_food_with_pheromone.lib.objects.food import ReFood
from scheme.pushing_food_with_pheromone.lib.objects.nest import NestBuilder, Nest

from .prerulde import Settings

from .brain import Brain
from .obj_builders import create_robot_builders, create_food_builders


class Task(optimizer.MjcTaskInterface):
    def __init__(self, time_length: int, world: World, nest: Nest, robot: Robot, refood: ReFood):
        self.time_length: int = time_length
        self.world: World = world
        self.nest: Nest = nest
        self.robot: Robot = robot
        self.refood: ReFood = refood
        self.dump = []

    def get_model(self) -> mujoco.MjModel:
        return self.world.model

    def get_data(self) -> mujoco.MjData:
        return self.world.data

    def calc_loss(self) -> float:
        vec_fn = self.refood.position - self.nest.position
        distance_fn = np.sum(
            np.maximum(np.linalg.norm(vec_fn, axis=1) - self.nest.size, 0)
        )
        loss_fn = -Settings.LOSS_RATE_FN * np.exp(-(distance_fn ** 2) / (Settings.LOSS_SIGMA_FN ** 2))

        vec_fr = self.refood.position - self.robot.position[0:2]
        distance_fr = np.sum(
            np.maximum(np.linalg.norm(vec_fr, axis=1) - self.robot.size - self.refood.size, 0)
        )
        loss_fr = -Settings.LOSS_RATE_FR * np.exp(-(distance_fr ** 2) / (Settings.LOSS_SIGMA_FR ** 2))

        self.dump.append((loss_fn, loss_fr))
        return loss_fn + loss_fr

    def calc_step(self) -> float:
        self.robot.action()
        self.world.calc_step()
        return self.calc_loss()

    def get_dump_data(self) -> object | None:
        return self.dump

    def run(self) -> float:
        loss = 0
        for _ in range(self.time_length):
            loss += self.calc_step()
        return loss


class TaskGenerator(optimizer.TaskGenerator):
    def __init__(self):
        invalid_area = []
        self.robot_builders = [
            create_robot_builders(0, invalid_area)
        ]
        self.food_builders = [
            create_food_builders(0, invalid_area)
        ]

    def generate(self, para, debug=False) -> Task:
        brain = Brain(para, Settings.ROBOT_THINK_INTERVAL)
        for builder in self.robot_builders:
            builder.brain = brain
        builders = self.robot_builders + self.food_builders

        w_builder = WorldBuilder1(
            Settings.SIMULATION_TIMESTEP,
            (Settings.RENDER_WIDTH, Settings.RENDER_HEIGHT),
            Settings.WORLD_WIDTH, Settings.WORLD_HEIGHT
        )

        w_builder.add_builder(NestBuilder(
            Settings.NEST_POS, Settings.NEST_SIZE
        ))

        w_builder.add_builders(builders)

        world, w_objs = w_builder.build()

        robot = w_objs[self.robot_builders[0].builder_name]
        food = w_objs[self.food_builders[0].builder_name]
        nest = w_objs["nest_builder"]

        refood = ReFood(Settings.WORLD_WIDTH, Settings.WORLD_HEIGHT, w_builder.thickness, [food], [nest], [robot])

        return Task(Settings.SIMULATION_TIME_LENGTH, world, nest, robot, refood)
