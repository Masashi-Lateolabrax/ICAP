import enum
import tkinter as tk
import random

import mujoco
import numpy as np

from mujoco_xml_generator.utils import MuJoCoView
from mujoco_xml_generator import Sensor, Actuator, Body, WorldBody
from mujoco_xml_generator import common, body, actuator, asset

from scheme.pushing_food_with_pheromone.lib.world import WorldObjectBuilder, BaseWorldBuilder, WorldClock

TIMESTEP = 0.01
WIDTH = 5
HEIGHT = 5
MOVE_SPEED = 0.8
ANGULAR_SPEED = 3.14 / 2


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
    class State(enum.Enum):
        STOP = 0
        GO_FRONT = 1
        GO_BACK = 2
        TURN_LEFT = 4
        TURN_RIGHT = 8

        def __str__(self):
            res = "stop"
            match self:
                case self.GO_BACK:
                    res = "back"
                case self.GO_FRONT:
                    res = "front"
                case self.TURN_LEFT:
                    res = "left"
                case self.TURN_RIGHT:
                    res = "right"
            return res

    def __init__(self, body_, act_x, act_y, act_r):
        from mujoco._structs import _MjDataBodyViews, _MjDataActuatorViews

        self.body: _MjDataBodyViews = body_
        self.act_x: _MjDataActuatorViews = act_x
        self.act_y: _MjDataActuatorViews = act_y
        self.act_r: _MjDataActuatorViews = act_r

        self.state = Robot.State.STOP

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
        r = random.uniform(0, 4 * 200)
        if 0 <= r < 4:
            self.state = Robot.State(2 ** int(r))
            print(self.state)

        match self.state:
            case Robot.State.GO_FRONT:
                v = self._cache["direction"]
                self.act_x.ctrl[0] = v[0] * MOVE_SPEED
                self.act_y.ctrl[0] = v[1] * MOVE_SPEED
                self.act_r.ctrl[0] = 0

            case Robot.State.GO_BACK:
                v = self._cache["direction"]
                self.act_x.ctrl[0] = v[0] * -MOVE_SPEED
                self.act_y.ctrl[0] = v[1] * -MOVE_SPEED
                self.act_r.ctrl[0] = 0

            case Robot.State.TURN_RIGHT:
                self.act_x.ctrl[0] = 0
                self.act_y.ctrl[0] = 0
                self.act_r.ctrl[0] = ANGULAR_SPEED

            case Robot.State.TURN_LEFT:
                self.act_x.ctrl[0] = 0
                self.act_y.ctrl[0] = 0
                self.act_r.ctrl[0] = -ANGULAR_SPEED

            case Robot.State.STOP:
                self.act_x.ctrl[0] = 0
                self.act_y.ctrl[0] = 0
                self.act_r.ctrl[0] = 0

    @property
    def GET_SPEED(self):
        return self._cache["speed"]


class RobotBuilder(WorldObjectBuilder):
    def __init__(self, id_: int, pos: tuple[float, float, float]):
        super().__init__(f"robot{id_}_builder")
        self.name_table = {
            "body": f"robot{id_}_body",
            "joint_x": f"robot{id_}_joint_x",
            "joint_y": f"robot{id_}_joint_y",
            "joint_r": f"robot{id_}_joint_r",
            "act_x": f"robot{id_}_act_x",
            "act_y": f"robot{id_}_act_y",
            "act_r": f"robot{id_}_act_r",
        }
        self.pos = pos

    def gen_body(self) -> Body | None:
        return Body(
            name=self.name_table["body"], pos=self.pos,
        ).add_children([
            body.Geom(
                type_=common.GeomType.CYLINDER, size=(0.175, 0.05), rgba=(1, 1, 0, 0.5), condim=1
            ),

            body.Joint(
                name=self.name_table["joint_x"], type_=common.JointType.SLIDE, axis=(1, 0, 0)
            ),
            body.Joint(
                name=self.name_table["joint_y"], type_=common.JointType.SLIDE, axis=(0, 1, 0)
            ),
            body.Joint(
                name=self.name_table["joint_r"], type_=common.JointType.HINGE, axis=(0, 0, 1)
            ),

            body.Site(
                type_=common.GeomType.SPHERE, size=(0.04,), rgba=(1, 0, 0, 1), pos=(0, 0.13, 0.051),
            ),
            body.Site(type_=common.GeomType.SPHERE, size=(0.04,)),
        ])

    def gen_act(self) -> Actuator | None:
        return Actuator().add_children([
            actuator.Velocity(
                name=self.name_table["act_x"], joint=self.name_table["joint_x"], kv=100000
            ),
            actuator.Velocity(
                name=self.name_table["act_y"], joint=self.name_table["joint_y"], kv=100000
            ),
            actuator.Velocity(
                name=self.name_table["act_r"], joint=self.name_table["joint_r"], kv=1000
            )
        ])

    def gen_sen(self) -> Sensor | None:
        return None

    def extract(self, _model: mujoco.MjModel, data: mujoco.MjData, timer: WorldClock):
        body_ = data.body(self.name_table["body"])
        act_x = data.actuator(self.name_table["act_x"])
        act_y = data.actuator(self.name_table["act_y"])
        act_r = data.actuator(self.name_table["act_r"])
        return Robot(body_, act_x, act_y, act_r)


class App(tk.Tk):
    def __init__(self, width=640, height=480):
        super().__init__()

        self.title("World")

        self.frame = tk.Frame(self)
        self.frame.pack()

        self.view = MuJoCoView(self.frame, width, height)
        self.view.enable_input()
        self.view.pack()

        self.world, w_objs = WorldBuilder(TIMESTEP, (width, height), WIDTH, HEIGHT).add_builder(
            RobotBuilder(0, (0, 0, 0))
        ).build()

        self.robot: Robot = w_objs["robot0_builder"]

        self.renderer = mujoco.Renderer(
            self.world.model, height, width, 3000
        )
        self.after(0, self.update)

    def update(self):
        self.robot.update()
        self.robot.action()
        self.world.calc_step()

        self.view.render(self.world.data, self.renderer, dummy_geoms=self.world.get_dummy_geoms())

        self.after(1, self.update)


def main():
    app = App()
    app.mainloop()


if __name__ == '__main__':
    main()
