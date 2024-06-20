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


def gen_xml() -> str:
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
        ])
    ])

    worldbody = WorldBody().add_children([
        body.Geom(
            type_=mjc_cmn.GeomType.PLANE, pos=(0, 0, 0), size=(1, 1, 1), material="ground", rgba=(0, 0, 0, 0)
        ),
    ])
    act = Actuator()

    worldbody.add_children([
        body.Geom(
            name="nest", type_=mjc_cmn.GeomType.CYLINDER,
            pos=(HyperParameters.Environment.NEST_POS[0], HyperParameters.Environment.NEST_POS[1], 0.025),
            size=(HyperParameters.Environment.NEST_SIZE, 0.025), rgba=(0, 1, 0, 1), conaffinity=2, contype=2,
        )
    ])

    for i, p in enumerate(HyperParameters.Environment.FOOD_POS):
        worldbody.add_children([
            Body().add_children([
                body.Geom(
                    name=f"food{i}", type_=mjc_cmn.GeomType.CYLINDER, pos=(p[0], p[1], 0.071), size=(0.5, 0.07),
                    rgba=(0, 1, 1, 1), density=300000, condim=3
                ),
                body.Joint(
                    name=f"food{i}.joint.slide_x", type_=mjc_cmn.JointType.SLIDE, axis=(1, 0, 0),
                    frictionloss=3000000 * 3
                ),
                body.Joint(
                    name=f"food{i}.joint.slide_y", type_=mjc_cmn.JointType.SLIDE, axis=(0, 1, 0),
                    frictionloss=3000000 * 3
                )
            ]),
        ])

    bot_rev_quat = np.zeros((4, 1))
    act_dir = np.zeros((2, 3))
    for i, p in enumerate(HyperParameters.Environment.BOT_POS):
        mujoco.mju_axisAngle2Quat(bot_rev_quat, [0, 0, 1], -p[2] / 180 * mujoco.mjPI)
        mujoco.mju_rotVecQuat(act_dir[0], [1, 0, 0], bot_rev_quat)
        mujoco.mju_rotVecQuat(act_dir[1], [0, 1, 0], bot_rev_quat)

        worldbody.add_children([
            Body(
                name=f"bot{i}.body", pos=(p[0], p[1], 0.06),
                orientation=mjc_cmn.Orientation.AxisAngle(0, 0, 1, p[2])
            ).add_children([
                body.Geom(
                    name=f"bot{i}.geom", type_=mjc_cmn.GeomType.CYLINDER,
                    size=(0.175, 0.05), rgba=(1, 1, 0, 0.5), mass=30e3, condim=1
                ),

                body.Joint(
                    name=f"bot{i}.joint.slide_x", type_=mjc_cmn.JointType.SLIDE, axis=act_dir[0]
                ),
                body.Joint(
                    name=f"bot{i}.joint.slide_y", type_=mjc_cmn.JointType.SLIDE, axis=act_dir[1]
                ),
                body.Joint(name=f"bot{i}.joint.hinge", type_=mjc_cmn.JointType.HINGE, axis=(0, 0, 1)),

                body.Site(
                    type_=mjc_cmn.GeomType.SPHERE, size=(0.04,), rgba=(1, 0, 0, 1), pos=(0, 0.13, 0.051),
                ),
                body.Site(name=f"bot{i}.site.center", type_=mjc_cmn.GeomType.SPHERE, size=(0.04,)),

                body.Camera(
                    f"bot{i}.camera", pos=(0, 0.176, 0),
                    orientation=mjc_cmn.Orientation.AxisAngle(1, 0, 0, 90)
                ),
                body.Camera(
                    f"bot{i}.camera_top", pos=(0, 0, 2),
                )
            ])
        ])
        act.add_children([
            actuator.Velocity(
                name=f"bot{i}.act.pos_x", joint=f"bot{i}.joint.slide_x",
                kv=10000000, forcelimited=mjc_cmn.BoolOrAuto.TRUE, forcerange=(-3000000, 3000000)
            ),
            actuator.Velocity(
                name=f"bot{i}.act.pos_y", joint=f"bot{i}.joint.slide_y",
                kv=10000000, forcelimited=mjc_cmn.BoolOrAuto.TRUE, forcerange=(-3000000, 3000000)
            ),
            actuator.Velocity(
                name=f"bot{i}.act.rot", joint=f"bot{i}.joint.hinge",
                kv=10000000
            ),
        ])

    generator.add_children([
        worldbody,
        act
    ])

    xml = generator.build()
    return xml


if __name__ == '__main__':
    print(gen_xml())
