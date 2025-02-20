import tkinter as tk
import mujoco

from mujoco_xml_generator.utils import MuJoCoView

from scheme.pushing_food_with_pheromone.lib.world import WorldBuilder1
from scheme.pushing_food_with_pheromone.lib.objects.food import Food, FoodBuilder, ReFood
from scheme.pushing_food_with_pheromone.lib.objects.robot import RobotBuilder, BrainInterface, BrainJudgement, Robot
from scheme.pushing_food_with_pheromone.lib.objects.nest import NestBuilder, Nest

RENDER_WIDTH = 500
RENDER_HEIGHT = 500
MAX_GEOM = 3000

WORLD_WIDTH = 7
WORLD_HEIGHT = 7

ROBOT_SIZE = 0.175
ROBOT_WEIGHT = 30  # kg
ROBOT_MOVE_SPEED = 0.8
ROBOT_TURN_SPEED = 3.14 / 2

FOOD_SIZE = 0.5
FOOD_DENSITY = 500
FOOD_FRICTIONLOSS = 2

NEST_SIZE = 1.0

SENSOR_GAIN = 1
SENSOR_OFFSET = ROBOT_SIZE + FOOD_SIZE


class Brain(BrainInterface):
    def __init__(self):
        self.state = BrainJudgement.STOP

    @staticmethod
    def get_dim() -> int:
        return 0

    def set_state(self, state: BrainJudgement):
        self.state = state

    def think(self, input_) -> BrainJudgement:
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

        self.view.camera.lookat[:] = [0, 0, 0]
        self.view.camera.distance = 7
        self.view.camera.elevation = -90

        self.world, self.robot_brain, self.robot, self.refood = self._setup_objects(width, height)

        self.bind("<KeyPress-w>", lambda e: self.robot_brain.set_state(BrainJudgement.FORWARD))
        self.bind("<KeyPress-s>", lambda e: self.robot_brain.set_state(BrainJudgement.BACK))
        self.bind("<KeyPress-a>", lambda e: self.robot_brain.set_state(BrainJudgement.TURN_LEFT))
        self.bind("<KeyPress-d>", lambda e: self.robot_brain.set_state(BrainJudgement.TURN_RIGHT))
        self.bind("<KeyRelease-w>", lambda e: self.robot_brain.set_state(BrainJudgement.STOP))
        self.bind("<KeyRelease-s>", lambda e: self.robot_brain.set_state(BrainJudgement.STOP))
        self.bind("<KeyRelease-a>", lambda e: self.robot_brain.set_state(BrainJudgement.STOP))
        self.bind("<KeyRelease-d>", lambda e: self.robot_brain.set_state(BrainJudgement.STOP))

        self.renderer = mujoco.Renderer(
            self.world.model, height, width, 3000
        )

        self.world.calc_step()
        self.after(0, self.update)

    @staticmethod
    def _setup_objects(width, height):
        robot_brain = Brain()

        w_builder = WorldBuilder1(
            0.01, (width, height), WORLD_WIDTH, WORLD_HEIGHT
        ).add_builder(
            NestBuilder((0, 0), NEST_SIZE)
        ).add_builder(
            FoodBuilder(0, (0, 1.5), FOOD_SIZE, FOOD_DENSITY, FOOD_FRICTIONLOSS)
        ).add_builder(RobotBuilder(
            0,
            robot_brain,
            (0, -1.5, 0),
            ROBOT_SIZE, ROBOT_WEIGHT, ROBOT_MOVE_SPEED, ROBOT_TURN_SPEED, SENSOR_GAIN, SENSOR_OFFSET,
            1
        ))

        world, w_objs = w_builder.build()

        nest: Nest = w_objs["nest_builder"]
        robot: Robot = w_objs["robot0_builder"]
        food: Food = w_objs["food0_builder"]
        refood = ReFood(
            WORLD_WIDTH,
            WORLD_HEIGHT,
            w_builder.thickness,
            [food], [nest], [robot]
        )

        return world, robot_brain, robot, refood

    def update(self):
        self.robot.action()
        self.refood.update()
        self.world.calc_step()

        self.view.render(self.world.data, self.renderer, dummy_geoms=self.world.get_dummy_geoms())

        self.after(1, self.update)


def main():
    app = App()
    app.mainloop()


if __name__ == '__main__':
    main()
