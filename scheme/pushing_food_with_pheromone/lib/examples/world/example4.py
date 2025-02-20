import tkinter as tk
import mujoco

from mujoco_xml_generator.utils import MuJoCoView

from scheme.pushing_food_with_pheromone.lib.world import WorldBuilder1
from scheme.pushing_food_with_pheromone.lib.objects.food import Food, FoodBuilder
from scheme.pushing_food_with_pheromone.lib.objects.robot import RobotBuilder, BrainInterface, BrainJudgement, Robot

RENDER_WIDTH = 500
RENDER_HEIGHT = 500
MAX_GEOM = 3000

WORLD_WIDTH = 5
WORLD_HEIGHT = 5

ROBOT_SIZE = 0.175
ROBOT_WEIGHT = 30  # kg
ROBOT_MOVE_SPEED = 0.8
ROBOT_TURN_SPEED = 3.14 / 4
ROBOT_THINK_INTERVAL = 10

FOOD_SIZE = 0.5
FOOD_FRICTIONLOSS = 1500

SENSOR_GAIN = 1
SENSOR_OFFSET = ROBOT_SIZE + FOOD_SIZE


class Brain(BrainInterface):
    @staticmethod
    def get_dim() -> int:
        return 0

    def think(self, input_) -> BrainJudgement:
        return BrainJudgement.TURN_LEFT


class App(tk.Tk):
    def __init__(self, width=640, height=480):
        super().__init__()

        self.title("World")

        self.frame = tk.Frame(self)
        self.frame.pack()

        self.view = MuJoCoView(self.frame, width, height)
        self.view.enable_input()
        self.view.pack()

        w_builder = WorldBuilder1(
            0.01, (width, height), WORLD_WIDTH, WORLD_HEIGHT
        ).add_builder(
            FoodBuilder(0, (0, 1.5), FOOD_SIZE, FOOD_FRICTIONLOSS)
        ).add_builder(
            RobotBuilder(
                0,
                Brain(),
                (0, -1.5, 90),
                ROBOT_SIZE, ROBOT_WEIGHT, ROBOT_MOVE_SPEED, ROBOT_TURN_SPEED, SENSOR_GAIN, SENSOR_OFFSET,
                1
            )
        )
        self.world, w_objs = w_builder.build()

        self.robot: Robot = w_objs["robot0_builder"]
        self.food: Food = w_objs["food0_builder"]

        self.renderer = mujoco.Renderer(
            self.world.model, height, width, 3000
        )
        self.after(0, self.update)

    def update(self):
        self.robot.action()
        self.world.calc_step()

        self.view.render(self.world.data, self.renderer, dummy_geoms=self.world.get_dummy_geoms())

        print(f"Local Direction: {self.robot.local_direction}")
        print(f"Global Direction: {self.robot.global_direction}")
        print(f"Relative position: {self.robot.calc_relative_position(self.food.position)}")
        print(f"Sensor value: {self.robot.input.get_food()}")
        print("")

        self.after(500, self.update)


def main():
    app = App()
    app.mainloop()


if __name__ == '__main__':
    main()
