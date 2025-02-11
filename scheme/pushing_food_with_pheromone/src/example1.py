import tkinter as tk
import mujoco
import numpy as np

from mujoco_xml_generator.utils import MuJoCoView
from mujoco_xml_generator import Sensor, Actuator, Body, WorldBody
from mujoco_xml_generator import common, body, asset, sensor

from .prerude import world
from .utils import random_point_avoiding_invalid_areas
from .mujoco_objects.robot import RobotBuilder, Robot

RENDER_WIDTH = 10
RENDER_HEIGHT = 10

WORLD_WIDTH = 10
WORLD_HEIGHT = 10

ROBOT_SIZE = 1
ROBOT_WEIGHT = 1

FOOD_SIZE = 1


class WorldBuilder(world.BaseWorldBuilder):
    @staticmethod
    def _create_wall(world_body: WorldBody, width: float, height: float):
        """
        ワールドの境界となる壁を作成する。

        Args:
            world_body (WorldBody): ワールドのボディオブジェクト。
            width (float): ワールドの幅。
            height (float): ワールドの高さ。
        """
        for name, x, y, w, h in [
            ("wallN", 0, height * 0.5, width * 0.5, 0.5),
            ("wallS", 0, height * -0.5, width * 0.5, 0.5),
            ("wallW", width * 0.5, 0, 0.5, height * 0.5),
            ("wallE", width * -0.5, 0, 0.5, height * 0.5),
        ]:
            world_body.add_children([
                body.Geom(
                    name=name, type_=common.GeomType.BOX,
                    pos=(x, y, 0.1), size=(w, h, 0.1),
                    condim=1
                )
            ])

    def __init__(self, timestep: float, resolution: tuple[int, int], width: float, height: float):
        """
        WorldBuilderのコンストラクタ。

        Args:
            timestep (float): シミュレーションのタイムステップ。
            resolution (tuple[int, int]): シミュレーションの解像度（幅、高さ）。
            width (float): ワールドの幅。
            height (float): ワールドの高さ。
        """
        from mujoco_xml_generator import Option, Visual
        from mujoco_xml_generator import visual

        super().__init__()

        self.generator.add_children([
            Option(
                timestep=timestep,
                integrator=common.IntegratorType.IMPLICITFACT
            ),
            Visual().add_children([
                visual.Global(
                    offwidth=resolution[0],
                    offheight=resolution[1]
                )
            ]),
        ])

        self.asset.add_children([
            asset.Texture(
                name="simple_checker", type_=common.TextureType.TWO_DiM,
                builtin=common.TextureBuiltinType.CHECKER, width=256, height=256,
                rgb1=(1., 1., 1.), rgb2=(0.7, 0.7, 0.7)
            ),
            asset.Material(
                name="ground", texture="simple_checker",
                texrepeat=(int(width / 2), int(height / 2))
            )
        ])

        self.world_body.add_children([
            body.Geom(
                type_=common.GeomType.PLANE, material="ground", rgba=(0, 0, 0, 1),
                pos=(0, 0, 0), size=(width * 0.5, height * 0.5, 1)
            ),
        ])

        WorldBuilder._create_wall(self.world_body, width, height)


class Food:
    def __init__(self, body_, speed_sensor):
        from mujoco._structs import _MjDataBodyViews, _MjDataSensorViews

        self.body: _MjDataBodyViews = body_
        self.speed_sensor: _MjDataSensorViews = speed_sensor

    def get_speed(self):
        return np.linalg.norm(self.speed_sensor.data)


class FoodBuilder(WorldObjectBuilder):
    def __init__(self, pos: tuple[float, float, float]):
        super().__init__("food_builder")
        self.name_table = {
            "body": "food_body",
            "joint_y": "food_joint_y",
            "center_site": "food_site",
            "velocisensor": "food_vel_sen"
        }
        self.pos = pos

    def gen_body(self) -> Body | None:
        return Body(name=self.name_table["body"], pos=self.pos).add_children([
            body.Geom(
                type_=common.GeomType.CYLINDER, size=(0.5, 0.07),
                rgba=(0, 1, 1, 1), density=1000, condim=3
            ),
            body.Joint(
                name=self.name_table["joint_y"], type_=common.JointType.SLIDE, axis=(0, 1, 0),
                frictionloss=FOOD_FRICTIONLOSS
            ),
            body.Site(
                name=self.name_table["center_site"]
            )
        ])

    def gen_act(self) -> Actuator | None:
        return None

    def gen_sen(self) -> Sensor | None:
        return Sensor().add_children([
            sensor.Velocimeter(self.name_table["center_site"], self.name_table["velocisensor"])
        ])

    def extract(self, data: mujoco.MjData, timer: WorldClock):
        body_ = data.body(self.name_table["body"])
        speed_sensor = data.sensor(self.name_table["velocisensor"])
        return Food(body_, speed_sensor)


class App(tk.Tk):
    def __init__(self, width=640, height=480):
        super().__init__()

        self.title("World")

        self.frame = tk.Frame(self)
        self.frame.pack()

        self.view = MuJoCoView(self.frame, width, height)
        self.view.enable_input()
        self.view.pack()

        self.renderer = mujoco.Renderer(
            self.world.model, height, width, 3000
        )
        self.after(0, self.update)

    def update(self):
        for bot in self.robot:
            bot.action()
        self.world.calc_step()

        self.view.render(self.world.data, self.renderer, dummy_geoms=self.world.get_dummy_geoms())
        print(self.food.get_speed())

        self.after(1, self.update)


def init_simulator():
    invalid_area = []

    w_builder = WorldBuilder(
        0.01,
        (RENDER_WIDTH, RENDER_HEIGHT),
        WORLD_WIDTH, WORLD_HEIGHT
    )

    # Create Robot
    pos = random_point_avoiding_invalid_areas(
        (WORLD_WIDTH * -0.5, WORLD_HEIGHT * 0.5),
        (WORLD_WIDTH * 0.5, WORLD_HEIGHT * -0.5),
        invalid_area,
    )
    w_builder.add_builder(
        RobotBuilder(0, (pos[0], pos[1], 180), ROBOT_SIZE, ROBOT_WEIGHT)
    )
    invalid_area.append(
        np.array([pos[0], pos[1], ROBOT_SIZE])
    )

    # Create Food
    pos = random_point_avoiding_invalid_areas(
        (WORLD_WIDTH * -0.5, WORLD_HEIGHT * 0.5),
        (WORLD_WIDTH * 0.5, WORLD_HEIGHT * -0.5),
        invalid_area,
    )
    w_builder.add_builder(
        FoodBuilder(0, (pos[0], pos[1]), FOOD_SIZE)
    )
    invalid_area.append(
        np.array([pos[0], pos[1], FOOD_SIZE])
    )

    world_, w_objs = w_builder.build()
    robot_ = w_objs["robot0_builder"]
    food_ = w_objs["food_builder"]

    return world_, robot_, food_


def main():
    app = App()
    app.mainloop()


if __name__ == '__main__':
    main()
