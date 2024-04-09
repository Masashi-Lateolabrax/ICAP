import mujoco_xml_generator as mjc_gen
import mujoco_xml_generator.common as mjc_cmn

from mujoco_xml_generator import Option
from mujoco_xml_generator import WorldBody, Body, body
from mujoco_xml_generator import Visual, visual
from mujoco_xml_generator import Actuator, actuator


def gen_xml() -> str:
    generator = mjc_gen.Generator().add_children([
        Option(timestep=0.003),

        Visual().add_children([
            visual.Global(offwidth=1024, offheight=768)
        ]),

        WorldBody().add_children([
            body.Geom(
                type_=mjc_cmn.GeomType.PLANE, pos=(0, 0, 0), size=(100, 100, 1), rgba=(1, 1, 1, 1)
            ),
            body.Geom(
                type_=mjc_cmn.GeomType.BOX, pos=(0, 15, 0.5), size=(0.5, 0.5, 0.5), rgba=(1, 0, 0, 1)
            ),

            Body(name="bot1.body", pos=(0, 0, 0.06)).add_children([
                body.Joint(name="bot1.jointS", type_=mjc_cmn.JointType.SLIDE, axis=(0, 1, 0)),
                body.Joint(name="bot1.jointH", type_=mjc_cmn.JointType.HINGE),
                body.Geom(type_=mjc_cmn.GeomType.CYLINDER, size=(0.3, 0.05), rgba=(1, 1, 0, 1), mass=30e3),
                body.Site(
                    type_=mjc_cmn.GeomType.SPHERE, size=(0.05,), rgba=(1, 0, 0, 1), pos=(0, 0.2, 0.051)
                ),
                body.Camera("bot1.camera1", pos=(0, 0, 5.))
            ]),
        ]),

        Actuator().add_children([
            actuator.Velocity(
                name="bot1.act.move", joint="bot1.jointS",
                forcelimited=mjc_cmn.BoolOrAuto.TRUE, forcerange=(-90e3 * 0.7, 90e3 * 1.5),
                ctrllimited=mjc_cmn.BoolOrAuto.TRUE, ctrlrange=(-0.7, 1.5),
                kv=90e3 * 1.5
            ),
            actuator.Velocity(
                name="bot1.act.rot", joint="bot1.jointH",
                ctrllimited=mjc_cmn.BoolOrAuto.TRUE, ctrlrange=(-1.5, 1.5),
                kv=100
            ),
        ])
    ])
    xml = generator.build()
    return xml


if __name__ == '__main__':
    print(gen_xml())
