import mujoco_xml_generator as mjc_gen
import mujoco_xml_generator.common as mjc_cmn

from mujoco_xml_generator import Option
from mujoco_xml_generator import WorldBody, Body, body
from mujoco_xml_generator import Visual, visual
from mujoco_xml_generator import Actuator, actuator
from mujoco_xml_generator import Asset, asset


def gen_xml(bots: list[tuple[float, float, float]], goals: list[tuple[float, float, float]]) -> str:
    generator = mjc_gen.Generator().add_children([
        Option(timestep=0.01, impratio=10, noslip_iterations=5),
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
        ])
    ])

    worldbody = WorldBody().add_children([
        body.Geom(
            type_=mjc_cmn.GeomType.PLANE, pos=(0, 0, 0), size=(5, 5, 1), material="ground",
        ),
    ])
    act = Actuator()

    for i, p in enumerate(goals):
        worldbody.add_children([
            body.Geom(
                name=f"goal{i}", type_=mjc_cmn.GeomType.CYLINDER, pos=(p[0], p[1], 0.025), size=(0.4, 0.025),
                rgba=(0, 1, 0, 1), conaffinity=2, contype=2,
            )
        ])

    for i, p in enumerate(bots):
        worldbody.add_children([
            Body(
                name=f"bot{i}.body", pos=(p[0], p[1], 0.06), orientation=mjc_cmn.Orientation.AxisAngle(0, 0, 1, p[2])
            ).add_children([
                body.Geom(
                    name=f"bot{i}.geom", type_=mjc_cmn.GeomType.CYLINDER,
                    size=(0.3, 0.05), rgba=(1, 1, 0, 0.5), mass=30e3,
                ),

                body.Joint(name=f"bot{i}.joint.slide_x", type_=mjc_cmn.JointType.SLIDE, axis=(1, 0, 0)),
                body.Joint(name=f"bot{i}.joint.slide_y", type_=mjc_cmn.JointType.SLIDE, axis=(0, 1, 0)),
                body.Joint(name=f"bot{i}.joint.hinge", type_=mjc_cmn.JointType.HINGE, axis=(0, 0, 1)),

                body.Site(
                    type_=mjc_cmn.GeomType.SPHERE, size=(0.05,), rgba=(1, 0, 0, 1), pos=(0, 0.2, 0.051),
                ),
                body.Site(name=f"bot{i}.site.center", type_=mjc_cmn.GeomType.SPHERE, size=(0.05,)),

                body.Camera(
                    f"bot{i}.camera", pos=(0, 0.31, 0),
                    orientation=mjc_cmn.Orientation.AxisAngle(1, 0, 0, 90)
                ),
                body.Camera(
                    f"bot{i}.camera_top", pos=(0, 0, 2),
                )
            ])
        ])
        act.add_children([
            actuator.Position(
                name=f"bot{i}.act.pos_x", joint=f"bot{i}.joint.slide_x",
                kp=30000000, kv=1000000
            ),
            actuator.Position(
                name=f"bot{i}.act.pos_y", joint=f"bot{i}.joint.slide_y",
                kp=30000000, kv=1000000
            ),
            actuator.Position(
                name=f"bot{i}.act.rot", joint=f"bot{i}.joint.hinge",
                kp=1000000, kv=100000
            ),
        ])

    generator.add_children([
        worldbody,
        act
    ])

    xml = generator.build()
    return xml


if __name__ == '__main__':
    print(gen_xml([(0, 0, 0), (10, 0, 0)], []))
