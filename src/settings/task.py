import mujoco

from src.optimizer import TaskInterface
from .hyper_parameters import HyperParameters
from .utils.robot import Robot

import mujoco_xml_generator as mjc_gen
import mujoco_xml_generator.common as mjc_cmn

from mujoco_xml_generator import Option
from mujoco_xml_generator import WorldBody, Body, body
from mujoco_xml_generator import Visual, visual
from mujoco_xml_generator import Actuator, actuator
from mujoco_xml_generator import Asset, asset


def gen_xml() -> str:
    generator = mjc_gen.Generator().add_children([
        Option(
            timestep=HyperParameters.Simulator.TIMESTEP,
            integrator=mjc_cmn.IntegratorType.IMPLICITFACT
        ),
        Visual().add_children([
            visual.Global(offwidth=500, offheight=500)
        ]),
        Asset().add_children([
            asset.Texture(
                name="simple_checker", type_=mjc_cmn.TextureType.TWO_DiM, builtin=mjc_cmn.TextureBuiltinType.CHECKER,
                width=100, height=100, rgb1=(1., 1., 1.), rgb2=(0.7, 0.7, 0.7)
            ),
            asset.Material(
                name="ground", texture="simple_checker", texrepeat=(10, 10)
            )
        ]),
        WorldBody().add_children([
            body.Geom(
                type_=mjc_cmn.GeomType.PLANE, pos=(0, 0, 0), size=(10, 10, 1), material="ground",
            ),

            Body(name=f"bot.body", pos=(0, 0, 0.06)).add_children([
                body.Geom(
                    name=f"bot.geom", type_=mjc_cmn.GeomType.CYLINDER,
                    size=(0.3, 0.05), rgba=(1, 1, 0, 0.5), mass=30e3,
                ),

                body.Joint(name=f"bot.joint.slide_x", type_=mjc_cmn.JointType.SLIDE, axis=(1, 0, 0)),
                body.Joint(name=f"bot.joint.slide_y", type_=mjc_cmn.JointType.SLIDE, axis=(0, 1, 0)),
                body.Joint(name=f"bot.joint.hinge", type_=mjc_cmn.JointType.HINGE, axis=(0, 0, 1)),

                body.Site(
                    type_=mjc_cmn.GeomType.SPHERE, size=(0.05,), rgba=(1, 0, 0, 1), pos=(0, 0.2, 0.051),
                ),
                body.Site(name=f"bot.site.center", type_=mjc_cmn.GeomType.SPHERE, size=(0.05,)),

                body.Camera(
                    f"bot.camera", pos=(0, 0.31, 0),
                    orientation=mjc_cmn.Orientation.AxisAngle(1, 0, 0, 90)
                ),
                body.Camera(
                    f"bot.camera_top", pos=(0, 0, 2),
                )
            ])
        ]),
        Actuator().add_children([
            actuator.Velocity(
                name=f"bot.act.pos_x", joint=f"bot.joint.slide_x",
                kv=10000000, forcelimited=mjc_cmn.BoolOrAuto.TRUE, forcerange=(-3000000, 3000000)
            ),
            actuator.Velocity(
                name=f"bot.act.pos_y", joint=f"bot.joint.slide_y",
                kv=10000000, forcelimited=mjc_cmn.BoolOrAuto.TRUE, forcerange=(-3000000, 3000000)
            ),
            actuator.Velocity(
                name=f"bot.act.rot", joint=f"bot.joint.hinge",
                kv=10000000
            ),
        ])
    ])

    xml = generator.build()
    return xml


class Task(TaskInterface):
    def __init__(self):
        xml = gen_xml()
        self._model: mujoco.MjModel = mujoco.MjModel.from_xml_string(xml)
        self._data: mujoco.MjData = mujoco.MjData(self._model)

        self._bot = Robot(self._model, self._data)

        self.counter = -1
        self.input_ = 0

    def get_model(self):
        return self._model

    def get_data(self):
        return self._data

    def calc_step(self) -> float:
        mujoco.mj_step(self._model, self._data)
        self.counter += 1

        if self.counter % int(HyperParameters.Simulator.TASK_INTERVAL / HyperParameters.Simulator.TIMESTEP) == 0:
            self.input_ += 1
            if self.input_ > 6:
                self.input_ = 0
            print(f"input: {self.input_}")

        self._bot.exec(self.input_)

        print(self._data.actuator_force)

        return 0.0

    def run(self) -> float:
        for _ in range(int(HyperParameters.Simulator.EPISODE / HyperParameters.Simulator.TIMESTEP + 0.5)):
            self.calc_step()
        return 0.0

    def get_bots(self):
        return [self._bot]
