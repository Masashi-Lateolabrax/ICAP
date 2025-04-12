from libs import optimizer
from libs.optimizer import Individual
import framework

from .settings import Settings
from .interfaceis import BrainBuilder, BrainInterface
from .task import Task

from .simulator.objects.utils import rand_robot_pos, rand_food_pos
from .simulator.objects import NestBuilder, RobotBuilder, FoodBuilder
from .simulator.world import WorldBuilder


class TaskGenerator(optimizer.TaskGenerator):
    def __init__(self, settings: Settings, brain_builder: BrainBuilder):
        self.settings = settings
        self.brain_builder = brain_builder

        invalid_area = []
        self.robot_positions = [
            rand_robot_pos(self.settings, invalid_area) for _ in range(settings.Robot.NUM)
        ]
        self.food_positions = [
            rand_food_pos(self.settings, invalid_area) for _ in range(settings.Food.NUM)
        ]

    def generate(self, para: Individual, debug=False) -> Task:
        robot_builders = [
            RobotBuilder(self.settings, i, self.brain_builder.build(para), pos_and_angle)
            for i, pos_and_angle in enumerate(self.robot_positions)
        ]
        food_builders = [
            FoodBuilder(i, pos) for i, pos in enumerate(self.food_positions)
        ]
        nest_builder = NestBuilder(self.settings)

        w_builder = WorldBuilder(self.settings)
        w_builder.add_builders(robot_builders)
        w_builder.add_builders(food_builders)
        w_builder.add_builder(nest_builder)

        world, w_objs = w_builder.build()

        robots = [w_objs[robot_builders[i].builder_name] for i in range(len(robot_builders))]
        food = [w_objs[food_builders[i].builder_name] for i in range(len(food_builders))]
        nest = w_objs[nest_builder.builder_name]

        refood = framework.simulator.objects.ReFood(
            Settings.Simulation.WORLD_WIDTH,
            Settings.Simulation.WORLD_HEIGHT,
            w_builder.thickness,
            food,
            nest,
            robots
        )

        return Task(self.settings, world, nest, robots, refood, debug)
