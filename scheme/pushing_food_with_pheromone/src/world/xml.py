import numpy as np
import mujoco

import mujoco_xml_generator as mjc_gen
import mujoco_xml_generator.common as mjc_cmn

from mujoco_xml_generator import Option
from mujoco_xml_generator import Visual, visual
from mujoco_xml_generator import Asset, asset
from mujoco_xml_generator import Body, WorldBody, body
from mujoco_xml_generator import Actuator, actuator

from ..settings import Settings
from ..utils import robot_names


def gen_xml(bot_pos: list[tuple[float, float, float]]) -> str:
    generator = mjc_gen.Generator().add_children([
        Option(
            timestep=Settings.Simulation.TIMESTEP,
            integrator=mjc_cmn.IntegratorType.IMPLICITFACT
        ),
        Visual().add_children([
            visual.Global(
                offwidth=Settings.Renderer.RESOLUTION[0],
                offheight=Settings.Renderer.RESOLUTION[1]
            )
        ]),
        Asset().add_children([
            asset.Texture(
                name="simple_checker", type_=mjc_cmn.TextureType.TWO_DiM, builtin=mjc_cmn.TextureBuiltinType.CHECKER,
                width=256, height=256,
                rgb1=(1., 1., 1.), rgb2=(0.7, 0.7, 0.7)
            ),
            asset.Material(
                name="ground", texture="simple_checker",
                texrepeat=(
                    int(Settings.Characteristic.Environment.WIDTH / 2),
                    int(Settings.Characteristic.Environment.HEIGHT / 2)
                )
            )
        ])
    ])

    tile_size = Settings.Characteristic.Environment.CELL_SIZE
    env_width = Settings.Characteristic.Environment.WIDTH * tile_size
    env_height = Settings.Characteristic.Environment.HEIGHT * tile_size

    worldbody = WorldBody().add_children([
        body.Geom(
            type_=mjc_cmn.GeomType.PLANE, material="ground", rgba=(0, 0, 0, 1),
            pos=(0, 0, 0), size=(env_width * 0.5, env_height * 0.5, 1)
        ),
    ])
    act = Actuator()

    for name, x, y, w, h in [
        ("wallN", 0, env_height * 0.5, env_width * 0.5, tile_size * 0.5),
        ("wallS", 0, env_height * -0.5, env_width * 0.5, tile_size * 0.5),
        ("wallW", env_width * 0.5, 0, tile_size * 0.5, env_height * 0.5),
        ("wallE", env_width * -0.5, 0, tile_size * 0.5, env_height * 0.5),
    ]:
        worldbody.add_children([
            body.Geom(
                name=name, type_=mjc_cmn.GeomType.BOX,
                pos=(x, y, 0.1), size=(w, h, 0.1),
                condim=1
            )
        ])

    worldbody.add_children([
        body.Geom(
            name="nest", type_=mjc_cmn.GeomType.CYLINDER,
            pos=(Settings.Task.Nest.POSITION[0], Settings.Task.Nest.POSITION[1], 0.025),
            size=(Settings.Task.Nest.SIZE, 0.025), rgba=(0, 1, 0, 1), conaffinity=2, contype=2,
        )
    ])

    for i, p in enumerate(Settings.Task.Food.POSITIONS):
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
    for i, p in enumerate(bot_pos):
        name_table = robot_names(i)

        mujoco.mju_axisAngle2Quat(bot_rev_quat, [0, 0, 1], -p[2] / 180 * mujoco.mjPI)
        mujoco.mju_rotVecQuat(act_dir[0], [1, 0, 0], bot_rev_quat)
        mujoco.mju_rotVecQuat(act_dir[1], [0, 1, 0], bot_rev_quat)

        worldbody.add_children([
            Body(
                name=name_table["body"], pos=(p[0], p[1], 0.06),
                orientation=mjc_cmn.Orientation.AxisAngle(0, 0, 1, p[2])
            ).add_children([
                body.Geom(
                    name=name_table["geom"], type_=mjc_cmn.GeomType.CYLINDER,
                    size=(0.175, 0.05), rgba=(1, 1, 0, 0.5), mass=30e3, condim=1
                ),

                body.Joint(
                    name=name_table["x_joint"], type_=mjc_cmn.JointType.SLIDE, axis=act_dir[0]
                ),
                body.Joint(
                    name=name_table["y_joint"], type_=mjc_cmn.JointType.SLIDE, axis=act_dir[1]
                ),
                body.Joint(name=name_table["r_joint"], type_=mjc_cmn.JointType.HINGE, axis=(0, 0, 1)),

                body.Site(
                    type_=mjc_cmn.GeomType.SPHERE, size=(0.04,), rgba=(1, 0, 0, 1), pos=(0, 0.13, 0.051),
                ),
                body.Site(name=name_table["c_site"], type_=mjc_cmn.GeomType.SPHERE, size=(0.04,)),

                body.Camera(
                    name_table["camera"], pos=(0, 0.176, 0),
                    orientation=mjc_cmn.Orientation.AxisAngle(1, 0, 0, 90)
                ),
            ])
        ])
        act.add_children([
            actuator.Velocity(
                name=name_table["x_act"], joint=name_table["x_joint"],
                kv=10000000, forcelimited=mjc_cmn.BoolOrAuto.TRUE, forcerange=(-3000000, 3000000)
            ),
            actuator.Velocity(
                name=name_table["y_act"], joint=name_table["y_joint"],
                kv=10000000, forcelimited=mjc_cmn.BoolOrAuto.TRUE, forcerange=(-3000000, 3000000)
            ),
            actuator.Velocity(
                name=name_table["r_act"], joint=name_table["r_joint"],
                kv=10000000
            ),
        ])

    generator.add_children([
        worldbody,
        act
    ])

    xml = generator.build()
    return xml
