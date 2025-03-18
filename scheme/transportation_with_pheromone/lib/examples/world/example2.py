import tkinter as tk
import random

import mujoco

from mujoco_xml_generator.utils import MuJoCoView

from scheme.transportation_with_pheromone.lib.world import WorldBuilder1
from scheme.transportation_with_pheromone.lib.objects.robot import RobotBuilder, BrainInterface, BrainJudgement, Robot

TIMESTEP = 0.01
WIDTH = 5
HEIGHT = 5

ROBOT_SIZE = 0.175
ROBOT_WEIGHT = 30  # kg
ROBOT_MOVE_SPEED = 0.8
ROBOT_TURN_SPEED = 3.14 / 2

SENSOR_GAIN = 1


class RobotBrain(BrainInterface):
    def __init__(self):
        self.state = BrainJudgement.STOP

    @staticmethod
    def get_dim() -> int:
        return 0

    def think(self, input_) -> BrainJudgement:
        r = random.uniform(0, 4 * 200)
        if 0 <= r < 4:
            self.state = BrainJudgement(int(r))
            print(self.state)
        return self.state


class App(tk.Tk):
    def __init__(self, width=640, height=480):
        super().__init__()

        self.title("World")

        self.frame = tk.Frame(self)
        self.frame.pack()

        self.view = MuJoCoView(self.frame, width, height)
        self.view.enable_input()
        self.view.pack()

        self.world, w_objs = WorldBuilder1(TIMESTEP, (width, height), WIDTH, HEIGHT).add_builder(
            RobotBuilder(
                0, RobotBrain(), (0, 0, 0), ROBOT_SIZE, ROBOT_WEIGHT, ROBOT_MOVE_SPEED, ROBOT_TURN_SPEED, SENSOR_GAIN,
                0, 0
            )
        ).build()

        self.robot: Robot = w_objs["robot0_builder"]

        self.renderer = mujoco.Renderer(
            self.world.model, height, width, 3000
        )
        self.after(0, self.update)

    def update(self):
        self.robot.action()
        self.world.calc_step()

        self.view.render(self.world.data, self.renderer, dummy_geoms=self.world.get_dummy_geoms())

        self.after(1, self.update)


def main():
    app = App()
    app.mainloop()


if __name__ == '__main__':
    main()
