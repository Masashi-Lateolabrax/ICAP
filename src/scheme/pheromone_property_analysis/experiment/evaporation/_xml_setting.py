from mujoco_xml_generator import common as mjc_cmn

from mujoco_xml_generator import Generator, Option
from mujoco_xml_generator import Visual, visual
from mujoco_xml_generator import Asset, asset
from mujoco_xml_generator import WorldBody, body

from .settings import Settings


def gen_xml():
    resolution = Settings.Display.RESOLUTION

    xml = Generator().add_children([
        Option(
            timestep=Settings.Simulation.TIMESTEP,
            integrator=mjc_cmn.IntegratorType.IMPLICITFACT
        ),
        Visual().add_children([
            visual.Global(offwidth=resolution[0], offheight=resolution[1])
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
                type_=mjc_cmn.GeomType.PLANE, pos=(0, 0, 0), size=(0, 0, 1), material="ground"
            ),
        ]),
    ]).build()
    return xml
