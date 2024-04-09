import mujoco_xml_generator as mjc_gen
import mujoco_xml_generator.common as mjc_cmn

from mujoco_xml_generator import Option
from mujoco_xml_generator import WorldBody, Body, body
from mujoco_xml_generator import Visual, visual


def gen_xml() -> str:
    generator = mjc_gen.Generator().add_children([
        Option(timestep=0.002),

        Visual().add_children([
            visual.Global(offwidth=1024, offheight=768)
        ]),

        WorldBody().add_children([
            body.Geom(
                type_=mjc_cmn.GeomType.PLANE, pos=(0, 0, 0), size=(100, 100, 1), rgba=(1, 1, 1, 1)
            ),

            Body(name="bot1.body", pos=(0, 0, 0.06)).add_children([
                body.Joint(type_=body.Joint.JointType.FREE),
                body.Geom(type_=mjc_cmn.GeomType.CYLINDER, size=(0.3, 0.05), rgba=(1, 1, 0, 1)),
                body.Camera("bot1.camera1", pos=(0, 0, 5.))
            ]),
        ])
    ])
    xml = generator.build()
    return xml


if __name__ == '__main__':
    print(gen_xml())
