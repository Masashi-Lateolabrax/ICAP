import tkinter as tk
import mujoco

from mujoco_xml_generator.utils import MuJoCoView
from mujoco_xml_generator import WorldBody
from mujoco_xml_generator import common, body, asset

from scheme.pushing_food_with_pheromone.lib.world import BaseWorldBuilder


class WorldBuilder(BaseWorldBuilder):
    @staticmethod
    def _create_wall(world_body: WorldBody, width: float, height: float):
        """
        ワールドの境界となる壁を作成する。

        Args:
            world_body (WorldBody): ワールドのボディオブジェクト。
            width (float): ワールドの幅。
            height (float): ワールドの高さ。
        """
        for name, x, y, w, h in [
            ("wallN", 0, height * 0.5, width * 0.5, 0.5),
            ("wallS", 0, height * -0.5, width * 0.5, 0.5),
            ("wallW", width * 0.5, 0, 0.5, height * 0.5),
            ("wallE", width * -0.5, 0, 0.5, height * 0.5),
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

        WorldBuilder._create_wall(self.world_body, width, height)


class App(tk.Tk):
    def __init__(self, width=640, height=480):
        super().__init__()

        self.title("World")

        self.frame = tk.Frame(self)
        self.frame.pack()

        self.view = MuJoCoView(self.frame, width, height)
        self.view.enable_input()
        self.view.pack()

        self.world, _ = WorldBuilder(
            0.01, (width, height), 30, 30
        ).build()

        self.renderer = mujoco.Renderer(
            self.world.model, height, width, 3000
        )

        self.after(0, self.update)

    def update(self):
        self.world.calc_step()
        self.view.render(self.world.data, self.renderer, dummy_geoms=self.world.get_dummy_geoms())
        self.after(1, self.update)


def main():
    app = App()
    app.mainloop()


if __name__ == '__main__':
    main()
