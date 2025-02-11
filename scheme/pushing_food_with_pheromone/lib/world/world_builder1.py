from mujoco_xml_generator import WorldBody
from mujoco_xml_generator import common, asset, body

from .src import BaseWorldBuilder


class WorldBuilder1(BaseWorldBuilder):
    @staticmethod
    def _create_wall(world_body: WorldBody, width: float, height: float):
        """
        ワールドの境界となる壁を作成する。

        Args:
            world_body (WorldBody): ワールドのボディオブジェクト。
            width (float): ワールドの幅。
            height (float): ワールドの高さ。
        """
        thickness = 0.5
        width += thickness * 2
        height += thickness * 2
        for name, x, y, w, h in [
            ("wallN", 0, height * 0.5, width * 0.5, thickness),
            ("wallS", 0, height * -0.5, width * 0.5, thickness),
            ("wallW", width * 0.5, 0, thickness, height * 0.5),
            ("wallE", width * -0.5, 0, thickness, height * 0.5),
        ]:
            world_body.add_children([
                body.Geom(
                    name=name, type_=common.GeomType.BOX,
                    pos=(x, y, 0.1), size=(w, h, 0.1),
                    condim=1
                )
            ])

    def __init__(self, timestep: float, resolution: tuple[int, int], width: float, height: float):
        """
        WorldBuilderのコンストラクタ。

        Args:
            timestep (float): シミュレーションのタイムステップ。
            resolution (tuple[int, int]): シミュレーションの解像度（幅、高さ）。
            width (float): ワールドの幅。
            height (float): ワールドの高さ。
        """
        from mujoco_xml_generator import Option, Visual
        from mujoco_xml_generator import visual

        super().__init__()

        self.generator.add_children([
            Option(
                timestep=timestep,
                integrator=common.IntegratorType.IMPLICITFACT
            ),
            Visual().add_children([
                visual.Global(
                    offwidth=resolution[0],
                    offheight=resolution[1]
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
                texrepeat=(int(width / 2), int(height / 2))
            )
        ])

        self.world_body.add_children([
            body.Geom(
                type_=common.GeomType.PLANE, material="ground", rgba=(0, 0, 0, 1),
                pos=(0, 0, 0), size=(width * 0.5, height * 0.5, 1)
            ),
        ])

        WorldBuilder1._create_wall(self.world_body, width, height)
