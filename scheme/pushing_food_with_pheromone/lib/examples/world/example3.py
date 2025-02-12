import tkinter as tk
import mujoco
import numpy as np

from mujoco_xml_generator.utils import MuJoCoView
from mujoco_xml_generator import Sensor, Actuator, Body, WorldBody
from mujoco_xml_generator import common, body, actuator, asset, sensor

from scheme.pushing_food_with_pheromone.lib.world import WorldObjectBuilder, BaseWorldBuilder, WorldClock

TIMESTEP = 0.01
NUM_ROBOTS = 3
ROBOT_WEIGHT = 30  # kg
ROBOT_SPEED = 0.8
FOOD_FRICTIONLOSS = 1500
WIDTH = 2.5
HEIGHT = 10


class WorldBuilder(BaseWorldBuilder):
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


class Robot:
    def __init__(self, body_, act_y):
        from mujoco._structs import _MjDataBodyViews, _MjDataActuatorViews

        self.body: _MjDataBodyViews = body_
        self.act_y: _MjDataActuatorViews = act_y

        self._cache = {
            "direction": np.array([0., 1, 0]),
            "position": np.zeros(3),
            "velocity": np.zeros(3),
            "speed": 0.0
        }

    def _update_direction(self):
        initial_direction = np.array([0., 1, 0])
        mujoco.mju_rotVecQuat(self._cache["direction"], initial_direction, self.body.xquat)
        return self._cache["direction"]

    def _update_movement(self):
        pos = np.copy(self.body.xpos)
        self._cache["velocity"] = (pos - self._cache["position"]) / TIMESTEP
        self._cache["position"] = pos
        self._cache["speed"] = float(np.linalg.norm(self._cache["velocity"]))

    def update(self):
        self._update_direction()
        self._update_movement()

    def action(self):
        self.act_y.ctrl[0] = ROBOT_SPEED

    @property
    def SPEED(self):
        return self._cache["speed"]


class RobotBuilder(WorldObjectBuilder):
    def __init__(self, id_: int, pos: tuple[float, float, float]):
        super().__init__(f"robot{id_}_builder")
        self.name_table = {
            "body": f"robot{id_}_body",
            "joint_y": f"robot{id_}_joint_y",
            "act_y": f"robot{id_}_act_y",
        }
        self.pos = pos

    def gen_body(self) -> Body | None:
        return Body(
            name=self.name_table["body"], pos=self.pos,
            orientation=common.Orientation.AxisAngle(0, 0, 1, 0)
        ).add_children([
            body.Geom(
                type_=common.GeomType.CYLINDER, size=(0.175, 0.05), rgba=(1, 1, 0, 0.5), mass=ROBOT_WEIGHT, condim=1
            ),

            body.Joint(
                name=self.name_table["joint_y"], type_=common.JointType.SLIDE, axis=(0, 1, 0)
            ),

            body.Site(
                type_=common.GeomType.SPHERE, size=(0.04,), rgba=(1, 0, 0, 1), pos=(0, 0.13, 0.051),
            ),
            body.Site(type_=common.GeomType.SPHERE, size=(0.04,)),
        ])

    def gen_act(self) -> Actuator | None:
        return Actuator().add_children([
            actuator.Velocity(
                name=self.name_table["act_y"], joint=self.name_table["joint_y"], kv=1000
            )
        ])

    def gen_sen(self) -> Sensor | None:
        return None

    def extract(self, model: mujoco.MjModel, data: mujoco.MjData, timer: WorldClock):
        body_ = data.body(self.name_table["body"])
        act_y = data.actuator(self.name_table["act_y"])
        return Robot(body_, act_y)


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

    def extract(self, _model: mujoco.MjModel, data: mujoco.MjData, timer: WorldClock):
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

        w_builder = WorldBuilder(0.01, (width, height), WIDTH, HEIGHT)

        w_builder.add_builder(FoodBuilder((0, 0, 0.071)))

        dw = (WIDTH - 1) / (NUM_ROBOTS + 1)
        for i in range(NUM_ROBOTS):
            w_builder.add_builder(RobotBuilder(
                i,
                (-0.5 * (WIDTH - 1) + dw * (i + 1), HEIGHT * -0.4, 0.1)
            ))

        self.world, w_objs = w_builder.build()

        self.robot: list[Robot] = [w_objs[f"robot{i}_builder"] for i in range(NUM_ROBOTS)]
        self.food: Food = w_objs["food_builder"]

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


def main():
    app = App()
    app.mainloop()


if __name__ == '__main__':
    main()
