from mujoco_xml_generator import Option, Visual
from mujoco_xml_generator import common, visual, asset, body

from libs.mujoco_builder import BaseWorldBuilder

from ..const import Settings
from ..objects.wall import WallBuilder


class WorldBuilder(BaseWorldBuilder):
    def __init__(self, settings: Settings):
        """
        WorldBuilderのコンストラクタ。

        Args:
            settings (Settings): シミュレーションの設定。
        """
        super().__init__()

        width = settings.Simulation.WORLD_WIDTH
        height = settings.Simulation.WORLD_HEIGHT

        self.generator.add_children([
            Option(
                timestep=settings.Simulation.TIME_STEP,
                integrator=common.IntegratorType.RK4
            ),
            Visual().add_children([
                visual.Global(
                    offwidth=settings.Simulation.Render.RENDER_WIDTH,
                    offheight=settings.Simulation.Render.RENDER_HEIGHT
                ),
                visual.HeadLight(
                    ambient=settings.Simulation.Render.LIGHT_AMBIENT,
                    diffuse=settings.Simulation.Render.LIGHT_DIFFUSE,
                    specular=settings.Simulation.Render.LIGHT_SPECULAR
                )
            ]),
        ])

        self.asset.add_children([
            asset.Texture(
                name="simple_checker", type_=common.TextureType.TWO_DiM,
                builtin=common.TextureBuiltinType.CHECKER, width=256, height=256,
                rgb1=(1., 1., 1.), rgb2=(0.7, 0.7, 0.7)
            ),
            asset.Material(
                name="ground", texture="simple_checker",
                texrepeat=(width / 2, height / 2)
            )
        ])

        self.world_body.add_children([
            body.Geom(
                type_=common.GeomType.PLANE, material="ground", rgba=(0, 0, 0, 0.5),
                pos=(0, 0, 0), size=(width * 0.5, height * 0.5, 1)
            ),
        ])

        self.thickness = 0.5
        self.add_builder(WallBuilder(width, height, self.thickness, 0.1))
