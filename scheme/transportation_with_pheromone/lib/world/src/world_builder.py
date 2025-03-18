from typing import Iterable

import mujoco

from mujoco_xml_generator import common, asset
from mujoco_xml_generator import Generator, WorldBody, Actuator, Sensor, Asset

from .world_clock import WorldClock
from .obj_builder import WorldObjectBuilder
from .world import World


class BaseWorldBuilder:
    """
    MuJoCOのシミュレータに渡すXMLの設計を支援する基底クラス。またWorldの生成も行う。

    # Functions

    - MuJoCoで使うXMLの設計
    - MuJoCoで使うXMLの生成
    - Worldの生成
    - XMLから生成されたMjDataを介してユーザの求める値を取得し，それをWorld生成時に返す

    # Details

    このクラスはMuJoCoで使うXMLの設計と生成を行う。XMLの設計はmujoco_xml_generatorを用いて行う．

    この基底クラスにはWorldの生成のための基本的なメソッドが定義されている．

    - add_texture(): テクスチャを登録する
    - add_material(): マテリアルを登録する
    - add_obj(): Worldにオブジェクトを追加する
    - build(): Worldを構築する

    またこのクラスを継承した子クラスでは次のメンバ変数が利用できる．

    - generator: mujoco_xml_generator.Generator
    - asset: mujoco_xml_generator.asset.Asset
    - world_body: mujoco_xml_generator.Body.WorldBody
    - sensor: mujoco_xml_generator.Sensor.Sensor
    - actuator: mujoco_xml_generator.Actuator.Actuator

    子クラスの__init__()内ではこれらのメンバ変数を用いてXMLの設計を行う．

    ```
    class MyWorldBuilder(BaseWorldBuilder):
        def __init__():
            super().__init__(self)
            self.world_body.add_children([
                body.Geom(
                    type_=common.GeomType.PLANE, pos=(0, 0, 0), size=(width * 0.5, height * 0.5, 1)
                ),
            ])
    ```
    """

    def __init__(self):
        self._objs: list[WorldObjectBuilder] = []
        self.generator = Generator()
        self.asset = Asset()
        self.world_body = WorldBody()
        self.sensor = Sensor()
        self.actuator = Actuator()

    def add_texture(
            self,
            name: str | None = None,
            type_: common.TextureType = common.TextureType.CUBE,
            content_type: str | None = None,
            file: str | None = None,
            gridsize: tuple[int, int] = (1, 1),
            gridlayout: str = "............",
            builtin: common.TextureBuiltinType = common.TextureBuiltinType.NONE,
            rgb1: tuple[float, float, float] = (0.8, 0.8, 0.8),
            rgb2: tuple[float, float, float] = (0.5, 0.5, 0.5),
            mark: common.TextureMark = common.TextureMark.NONE,
            markrgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
            random: float = 0.01,
            width: int = 0,
            height: int = 0,
            hflip: bool = False,
            vflip: bool = False
    ):
        """
        テクスチャを登録する．詳細はmujoco_xml_generator.asset.Textureを参照．
        """
        self.asset.add_children([
            asset.Texture(
                name=name, type_=type_, content_type=content_type, file=file,
                gridsize=gridsize, gridlayout=gridlayout, builtin=builtin,
                rgb1=rgb1, rgb2=rgb2, mark=mark, markrgb=markrgb,
                random=random, width=width, height=height, hflip=hflip, vflip=vflip
            )
        ])

    def add_material(
            self,
            name: str | None = None,
            class_: str | None = None,
            texture: str = None,
            texrepeat: tuple[float, float] = (1.0, 1.0),
            texuniform: bool = False,
            emission: float = 0,
            specular: float = 0.5,
            shininess: float = 0.5,
            reflectance: float = 0.0,
            rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    ):
        """
        マテリアルを登録する．詳細はmujoco_xml_generator.asset.Materialを参照．
        """
        self.asset.add_children([
            asset.Material(
                name=name, class_=class_, texture=texture, texrepeat=texrepeat,
                texuniform=texuniform, emission=emission, specular=specular,
                shininess=shininess, reflectance=reflectance, rgba=rgba
            )
        ])

    def add_builder(self, builder: WorldObjectBuilder):
        """
        Worldにオブジェクトを追加する。

        オブジェクトとはWorldObjectBuilderを継承したクラスであり，そのクラスを介してXMLに物体などの記述を加えられる．
        このメソッドでは渡されたWorldObjectBuilderの生成メソッドを呼び出し，その結果をXMLに追加する．

        Args:
            builder (WorldObjectBuilder): 追加するオブジェクトのビルダー。

        Returns:
            WorldBuilder: 自身のインスタンスを返す。
        """
        sg, b, a, s = builder._gen_all()
        if sg is not None and len(sg) > 0:
            self.world_body.add_children(sg)
        if b is not None:
            self.world_body.add_children([b])
        if a is not None:
            self.actuator.add_children(a.get_children())
        if s is not None:
            self.sensor.add_children(s.get_children())
        self._objs.append(builder)
        return self

    def add_builders(self, builders: Iterable[WorldObjectBuilder]):
        for builder in builders:
            self.add_builder(builder)

    def build(self) -> tuple[World, dict]:
        """
        Worldを構築する。

        Returns:
            tuple:
                - World: 生成結果のWorldインスタンス
                - dict: WorldObjectBuilderの名前をキーに持ち，そのWorldObjectBuilderが返したものが格納される
        """
        timer = WorldClock()

        self.generator.add_children([self.asset, self.world_body, self.actuator, self.sensor])
        xml = self.generator.build()
        model = mujoco.MjModel.from_xml_string(xml)
        world = World._create(model, timer)

        objects = {o.builder_name: o.extract(model, world.data, timer) for o in self._objs}

        return world, objects
