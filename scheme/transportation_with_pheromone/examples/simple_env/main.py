from scheme.transportation_with_pheromone.lib.objects.robot import Robot
from scheme.transportation_with_pheromone.lib.objects.food import Food
from scheme.transportation_with_pheromone.lib.world import WorldBuilder1

from scheme.transportation_with_pheromone.examples.simple_env.gui import App, SimulationManager

from obj_builders import create_robot_builders, create_food_builders
from brain import Brain
from settings import Settings


def init_simulator():
    w_builder = WorldBuilder1(
        0.01,
        (Settings.RENDER_WIDTH, Settings.RENDER_HEIGHT),
        Settings.WORLD_WIDTH, Settings.WORLD_HEIGHT
    )

    invalid_area = []
    brain = Brain()
    builders = {
        "robots": create_robot_builders(0, brain, invalid_area),
        "food": create_food_builders(0, invalid_area)
    }
    w_builder.add_builders(builders.values())

    world, w_objs = w_builder.build()
    robot: Robot = w_objs[builders["robots"].builder_name]
    food: Food = w_objs[builders["food"].builder_name]

    return world, robot, food


def main():
    world, robot, food = init_simulator()
    sim_manager = SimulationManager(world, robot, food)
    app = App(sim_manager, Settings.RENDER_WIDTH, Settings.RENDER_HEIGHT, 3000)
    app.mainloop()


if __name__ == '__main__':
    main()
