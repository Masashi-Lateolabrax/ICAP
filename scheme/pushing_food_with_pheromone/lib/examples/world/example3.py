import tkinter as tk
import mujoco
import numpy as np

from mujoco_xml_generator.utils import MuJoCoView

from scheme.pushing_food_with_pheromone.lib.world import WorldBuilder1
from scheme.pushing_food_with_pheromone.lib.objects.food import Food, FoodBuilder
from scheme.pushing_food_with_pheromone.lib.objects.robot import RobotBuilder, BrainInterface, BrainJudgement, Robot

RENDER_WIDTH = 500
RENDER_HEIGHT = 500
MAX_GEOM = 3000

WORLD_WIDTH = 2.5
WORLD_HEIGHT = 10

NUM_ROBOTS = 3
ROBOT_SIZE = 0.175
ROBOT_WEIGHT = 30  # kg
ROBOT_MOVE_SPEED = 0.8
ROBOT_TURN_SPEED = 3.14 / 2
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
        return BrainJudgement.FORWARD


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
            FoodBuilder(0, (0, 0), FOOD_SIZE, FOOD_FRICTIONLOSS)
        )

        dw = (WORLD_WIDTH - 1) / (NUM_ROBOTS + 1)
        for i in range(NUM_ROBOTS):
            w_builder.add_builder(RobotBuilder(
                i,
                Brain(),
                (-0.5 * (WORLD_WIDTH - 1) + dw * (i + 1), WORLD_HEIGHT * -0.4, 0.1),
                ROBOT_SIZE, ROBOT_WEIGHT, ROBOT_MOVE_SPEED, ROBOT_TURN_SPEED, SENSOR_GAIN, SENSOR_OFFSET,
                1
            ))

        self.world, w_objs = w_builder.build()

        self.robot: list[Robot] = [w_objs[f"robot{i}_builder"] for i in range(NUM_ROBOTS)]
        self.food: Food = w_objs["food0_builder"]

        self.renderer = mujoco.Renderer(
            self.world.model, height, width, 3000
        )
        self.after(0, self.update)

    def update(self):
        for bot in self.robot:
            bot.action()
        self.world.calc_step()

        self.view.render(self.world.data, self.renderer, dummy_geoms=self.world.get_dummy_geoms())
        print(np.linalg.norm(self.food.velocity))

        self.after(1, self.update)


def main():
    app = App()
    app.mainloop()


if __name__ == '__main__':
    main()
