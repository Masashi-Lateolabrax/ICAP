import numpy as np
import mujoco

import mujoco_xml_generator as mjc_gen
import mujoco_xml_generator.common as mjc_cmn

from mujoco_xml_generator import Option
from mujoco_xml_generator import Visual, visual
from mujoco_xml_generator import Asset, asset
from mujoco_xml_generator import Body, WorldBody, body
from mujoco_xml_generator import Actuator, actuator

from .hyper_parameters import HyperParameters


def gen_xml(bot_pos: tuple[float, float, float]) -> str:
    bot_rev_quat = np.zeros((4, 1))
    act_dir = np.zeros((2, 3))
    mujoco.mju_axisAngle2Quat(bot_rev_quat, [0, 0, 1], -bot_pos[2] / 180 * mujoco.mjPI)
    mujoco.mju_rotVecQuat(act_dir[0], [1, 0, 0], bot_rev_quat)
    mujoco.mju_rotVecQuat(act_dir[1], [0, 1, 0], bot_rev_quat)

    generator = mjc_gen.Generator().add_children([
        Option(
            timestep=HyperParameters.Simulator.TIMESTEP,
            integrator=mjc_cmn.IntegratorType.IMPLICITFACT
        ),
        Visual().add_children([
            visual.Global(
                offwidth=HyperParameters.Simulator.RESOLUTION[0],
                offheight=HyperParameters.Simulator.RESOLUTION[1]
            )
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
                type_=mjc_cmn.GeomType.PLANE, pos=(0, 0, 0), size=(5, 5, 1), material="ground",
            ),

            body.Geom(
                name=f"goal", type_=mjc_cmn.GeomType.CYLINDER,
                pos=(HyperParameters.Environment.GOAL_POS[0], HyperParameters.Environment.GOAL_POS[1], 0.025),
                size=(0.4, 0.025), rgba=(0, 1, 0, 1), conaffinity=2, contype=2,
            ),

            Body(
                name=f"bot.body", pos=(bot_pos[0], bot_pos[1], 0.06),
                orientation=mjc_cmn.Orientation.AxisAngle(0, 0, 1, bot_pos[2])
            ).add_children([
                body.Geom(
                    name="bot.geom", type_=mjc_cmn.GeomType.CYLINDER,
                    size=(0.3, 0.05), rgba=(1, 1, 0, 0.5), mass=30e3, condim=1
                ),

                body.Joint(name="bot.joint.slide_x", type_=mjc_cmn.JointType.SLIDE, axis=act_dir[0]),
                body.Joint(name="bot.joint.slide_y", type_=mjc_cmn.JointType.SLIDE, axis=act_dir[1]),
                body.Joint(name="bot.joint.hinge", type_=mjc_cmn.JointType.HINGE, axis=(0, 0, 1)),

                body.Site(
                    type_=mjc_cmn.GeomType.SPHERE, size=(0.05,), rgba=(1, 0, 0, 1), pos=(0, 0.2, 0.051),
                ),
                body.Site(name="bot.site.center", type_=mjc_cmn.GeomType.SPHERE, size=(0.05,)),

                body.Camera(
                    "bot.camera", pos=(0, 0.31, 0),
                    orientation=mjc_cmn.Orientation.AxisAngle(1, 0, 0, 90)
                ),
                body.Camera(
                    "bot.camera_top", pos=(0, 0, 2),
                )
            ]),
        ]),

        Actuator().add_children([
            actuator.Velocity(
                name="bot.act.pos_x", joint="bot.joint.slide_x",
                kv=10000000, forcelimited=mjc_cmn.BoolOrAuto.TRUE, forcerange=(-3000000, 3000000)
            ),
            actuator.Velocity(
                name="bot.act.pos_y", joint="bot.joint.slide_y",
                kv=10000000, forcelimited=mjc_cmn.BoolOrAuto.TRUE, forcerange=(-3000000, 3000000)
            ),
            actuator.Velocity(
                name="bot.act.rot", joint="bot.joint.hinge",
                kv=10000000
            ),
        ])
    ])

    xml = generator.build()
    return xml


if __name__ == '__main__':
    print(gen_xml())
