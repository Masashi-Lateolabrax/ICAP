import numpy as np
import mujoco

import mujoco_xml_generator as mjc_gen
import mujoco_xml_generator.common as mjc_cmn

from mujoco_xml_generator import Option
from mujoco_xml_generator import Visual, visual
from mujoco_xml_generator import Asset, asset
from mujoco_xml_generator import Body, WorldBody, body
from mujoco_xml_generator import Actuator, actuator

from .settings import Settings


def set_walls(worldbody: WorldBody):
    env_width, env_height = Settings.Environment.World.SIMULATION_FIELD_SIZE
    cell_size = Settings.Characteristics.Pheromone.CELL_SIZE_FOR_MUJOCO
    for name, x, y, w, h in [
        ("wallN", 0, env_height * 0.5, env_width * 0.5, cell_size * 0.5),
        ("wallS", 0, env_height * -0.5, env_width * 0.5, cell_size * 0.5),
        ("wallW", env_width * 0.5, 0, cell_size * 0.5, env_height * 0.5),
        ("wallE", env_width * -0.5, 0, cell_size * 0.5, env_height * 0.5),
    ]:
        worldbody.add_children([
            body.Geom(
                name=name, type_=mjc_cmn.GeomType.BOX,
                pos=(x, y, 0.1), size=(w, h, 0.1),
                condim=1
            )
        ])


def set_robots(worldbody: WorldBody, act: Actuator, bot_pos: list[tuple[float, float, float]]):
    bot_rev_quat = np.zeros((4, 1))
    act_dir = np.zeros((2, 3))

    eps = 0.00001
    robot_size = Settings.Environment.Robot.SIZE
    robot_height = robot_size * 0.3
    orientation_maker_size = robot_height * 0.75
    center_maker_size = robot_height * 0.75

    for i, p in enumerate(bot_pos):
        names = Settings.Environment.Robot.NAMES(i)

        mujoco.mju_axisAngle2Quat(bot_rev_quat, [0, 0, 1], -p[2] / 180 * mujoco.mjPI)
        mujoco.mju_rotVecQuat(act_dir[0], [1, 0, 0], bot_rev_quat)
        mujoco.mju_rotVecQuat(act_dir[1], [0, 1, 0], bot_rev_quat)

        worldbody.add_children([
            Body(
                name=names["body"], pos=(p[0], p[1], robot_height + eps),
                orientation=mjc_cmn.Orientation.AxisAngle(0, 0, 1, p[2])
            ).add_children([
                body.Geom(
                    name=names["geom"], type_=mjc_cmn.GeomType.CYLINDER,
                    size=(Settings.Environment.Robot.SIZE, robot_height), rgba=(1, 1, 0, 0.5), mass=30e3,
                    condim=1
                ),

                body.Joint(name=names["x_joint"], type_=mjc_cmn.JointType.SLIDE, axis=act_dir[0]),
                body.Joint(name=names["y_joint"], type_=mjc_cmn.JointType.SLIDE, axis=act_dir[1]),
                body.Joint(name=names["r_joint"], type_=mjc_cmn.JointType.HINGE, axis=(0, 0, 1)),

                body.Site(
                    type_=mjc_cmn.GeomType.SPHERE, size=(orientation_maker_size,), rgba=(1, 0, 0, 1),
                    pos=(0, robot_size - orientation_maker_size, robot_height),
                ),
                body.Site(type_=mjc_cmn.GeomType.SPHERE, size=(center_maker_size,)),

                body.Camera(
                    name=names["camera"], pos=(0, robot_size + eps, 0),
                    orientation=mjc_cmn.Orientation.AxisAngle(1, 0, 0, 90)
                ),
            ])
        ])
        act.add_children([
            actuator.Velocity(
                name=names["x_act"], joint=names["x_joint"],
                kv=Settings.Environment.Robot.Actuator.ACTUATOR_KV,
                forcelimited=mjc_cmn.BoolOrAuto.TRUE,
                forcerange=(
                    -Settings.Environment.Robot.Actuator.ACTUATOR_FORCE_RANGE,
                    Settings.Environment.Robot.Actuator.ACTUATOR_FORCE_RANGE
                )
            ),
            actuator.Velocity(
                name=names["y_act"], joint=names["y_joint"],
                kv=Settings.Environment.Robot.Actuator.ACTUATOR_KV,
                forcelimited=mjc_cmn.BoolOrAuto.TRUE,
                forcerange=(
                    -Settings.Environment.Robot.Actuator.ACTUATOR_FORCE_RANGE,
                    Settings.Environment.Robot.Actuator.ACTUATOR_FORCE_RANGE
                )
            ),
            actuator.Velocity(
                name=names["r_act"], joint=names["r_joint"],
                kv=Settings.Environment.Robot.Actuator.ACTUATOR_KV,
            ),
        ])


def set_safezone(worldbody: WorldBody, safe_zone_pos: list[tuple[float, float]]):
    for i, p in enumerate(safe_zone_pos):
        worldbody.add_children([
            body.Geom(
                name=f"safezone{i}", type_=mjc_cmn.GeomType.CYLINDER,
                pos=(p[0], p[1], Settings.Environment.SafeZone.SIZE[1]),
                size=Settings.Environment.SafeZone.SIZE,
                rgba=(0, 1, 0, 1),
                conaffinity=2, contype=2,
            ),
        ])


def gen_xml(
        timestep: float,
        safe_zone_pos: list[tuple[float, float]],
        bot_pos: list[tuple[float, float, float]]
) -> str:
    generator = mjc_gen.Generator().add_children([
        Option(
            timestep=timestep,
            integrator=mjc_cmn.IntegratorType.IMPLICITFACT
        ),
        Visual().add_children([
            visual.Global(
                offwidth=Settings.Display.RESOLUTION[0],
                offheight=Settings.Display.RESOLUTION[1]
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
            type_=mjc_cmn.GeomType.PLANE, pos=(0, 0, 0), size=(0, 0, 1), material="ground"
        ),
    ])
    act = Actuator()

    generator.add_children([worldbody, act])

    set_walls(worldbody)
    set_safezone(worldbody, safe_zone_pos)
    set_robots(worldbody, act, bot_pos)

    xml = generator.build()
    return xml


def test():
    timestep = Settings.Simulation.TIMESTEP
    safe_zone_pos = Settings.Task.SafeZone.POSITION()
    bot_pos = Settings.Task.Robot.POSITIONS()
    xml = gen_xml(timestep, safe_zone_pos, bot_pos)
    print(xml)
