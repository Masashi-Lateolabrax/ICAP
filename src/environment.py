import mujoco_xml_generator as mjc_gen
import mujoco_xml_generator.common as mjc_cmn

from mujoco_xml_generator import Option
from mujoco_xml_generator import WorldBody, Body, body
from mujoco_xml_generator import Visual, visual
from mujoco_xml_generator import Actuator, actuator


def gen_xml(bots: list[tuple[float, float, float]], goals: list[tuple[float, float, float]]) -> str:
    worldbody = WorldBody().add_children([
        body.Geom(
            type_=mjc_cmn.GeomType.PLANE, pos=(0, 0, 0), size=(5, 5, 1), rgba=(1, 1, 1, 1)
        ),
    ])
    act = Actuator()

    for i, p in enumerate(goals):
        worldbody.add_children([
            body.Geom(
                name=f"goal{i}", type_=mjc_cmn.GeomType.CYLINDER, pos=(p[0], p[1], 0.01), size=(0.4, 0.01),
                rgba=(0, 1, 0, 1), conaffinity=2, contype=2
            )
        ])

    for i, p in enumerate(bots):
        worldbody.add_children([
            Body(
                name=f"bot{i}.body", pos=(p[0], p[1], 0.06), orientation=mjc_cmn.Orientation.AxisAngle(0, 0, 1, p[2])
            ).add_children([
                body.Geom(type_=mjc_cmn.GeomType.CYLINDER, size=(0.3, 0.05), rgba=(1, 1, 0, 0.5), mass=30e3),

                body.Joint(type_=mjc_cmn.JointType.SLIDE, axis=(1, 0, 0)),
                body.Joint(type_=mjc_cmn.JointType.SLIDE, axis=(0, 1, 0)),
                body.Joint(name=f"bot{i}.joint.hinge", type_=mjc_cmn.JointType.HINGE),

                body.Site(
                    type_=mjc_cmn.GeomType.SPHERE, size=(0.05,), rgba=(1, 0, 0, 1), pos=(0, 0.2, 0.051)
                ),
                body.Site(name=f"bot{i}.site.center", type_=mjc_cmn.GeomType.SPHERE, size=(0.05,)),

                body.Camera(f"bot{i}.camera1", pos=(0, 0, 5.))
            ])
        ])
        act.add_children([
            actuator.Velocity(
                name=f"bot{i}.act.rot", joint=f"bot{i}.joint.hinge",
                ctrllimited=mjc_cmn.BoolOrAuto.TRUE, ctrlrange=(-1.5, 1.5),
                kv=1e5
            ),
            actuator.Velocity(
                name=f"bot{i}.act.move", site=f"bot{i}.site.center",
                gear=(0, 1, 0, 0, 0, 0),
                kv=1e5
            )
        ])

    generator = mjc_gen.Generator().add_children([
        Option(timestep=0.003),
        Visual().add_children([
            visual.Global(offwidth=1024, offheight=768)
        ]),
        worldbody,
        act
    ])

    xml = generator.build()
    return xml


if __name__ == '__main__':
    print(gen_xml([(0, 0), (10, 0)]))
