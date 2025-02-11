import mujoco
import numpy as np

from libs import optimizer

from scheme.pushing_food_with_pheromone.lib.world import World, WorldBuilder1
from scheme.pushing_food_with_pheromone.lib.objects.robot import Robot
from scheme.pushing_food_with_pheromone.lib.objects.food import Food

from .prerulde import Settings

from .brain import Brain
from .obj_builders import create_robot_builders, create_food_builders


class Task(optimizer.MjcTaskInterface):
    def __init__(self, time_length: int, world: World, robot: Robot, food: Food):
        self.time_length: int = time_length
        self.world: World = world
        self.robot: Robot = robot
        self.food: Food = food

    def get_model(self) -> mujoco.MjModel:
        return self.world.model

    def get_data(self) -> mujoco.MjData:
        return self.world.data

    def calc_loss(self) -> float:
        return np.linalg.norm(self.robot.position[0:2] - self.food.position)

    def calc_step(self) -> float:
        self.robot.action()
        self.world.calc_step()
        return self.calc_loss()

    def get_dump_data(self) -> object | None:
        pass

    def run(self) -> float:
        loss = 0
        for _ in range(self.time_length):
            loss += self.calc_step()
        return loss


class TaskGenerator(optimizer.TaskGenerator):
    def generate(self, para, debug=False) -> Task:
        brain = Brain(para, Settings.ROBOT_THINK_INTERVAL)

        invalid_area = []
        builders = {
            "robots": create_robot_builders(0, brain, invalid_area),
            "food": create_food_builders(0, invalid_area)
        }
        w_builder = WorldBuilder1(
            Settings.SIMULATION_TIMESTEP,
            (Settings.RENDER_WIDTH, Settings.RENDER_HEIGHT),
            Settings.WORLD_WIDTH, Settings.WORLD_HEIGHT
        )
        w_builder.add_builders(builders.values())

        world, w_objs = w_builder.build()

        robot = w_objs[builders["robots"].builder_name]
        food = w_objs[builders["food"].builder_name]

        return Task(Settings.SIMULATION_TIME_LENGTH, world, robot, food)
