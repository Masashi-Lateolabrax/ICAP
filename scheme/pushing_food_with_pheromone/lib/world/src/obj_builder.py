import abc

import mujoco
from mujoco_xml_generator import Body, Actuator, Sensor

from .world_clock import WorldClock


class WorldObjectBuilder(metaclass=abc.ABCMeta):
    """
    MuJoCoのシミュレーション上の物体を構築するための抽象基底クラス。

    # Functions

    - MjModelを生成するためのXMLにシミュレーション上の物体に関わる記述を加える
    - MjDataから必要なデータを取り出す

    # Details

    WorldBuilder::add_obj()でWorldObjectBuilderを追加すると，WorldBuilder::add_obj()の内部で

    - WorldObjectBuilder::gen_body()
    - WorldObjectBuilder::gen_act()
    - WorldObjectBuilder::gen_sen()

    が呼び出されます．これらのメソッドが返すBody, Actuator, SensorはWorldBuilderが保有するgeneratorに追加されます．
    これによってMuJoCoシミュレータ上の物体についてのXMLを追記することができます．

    そのため上記の3つのメソッドには生成したい物体のXMLを生成する実装を行ってください．

    追加された各WorldObjectBuilderのextract()はWorldBuilder::build()の内部で実行されます．
    WorldObjectBuilder::extract()はMjDataから適切な値の参照を得るためのメソッドです．

    WorldObjectBuilder::extract()の返り値は任意の型を指定可能です．

    ```
    class Human(WorldObjectBuilder):
        def __init__():
            super().__init__("human_factory")

        def gen_body(self):
            // Humanオブジェクトの構造を生成
            return Body(name="human").add_children([
                body.Geom(
                    type_=common.GeomType.CYLINDER, size=(0.25, 1.5)
                ),
                body.Joint(
                    name="joint_x", type_=common.JointType.SLIDE, axis=(1, 0, 0)
                ),
                body.Joint(
                    name="joint_y", type_=common.JointType.SLIDE, axis=(0, 1, 0)
                ),
            ])

        def gen_act():
            return None

        def gen_sen():
            return None

        def extract(self, data: MjData):
            return {
                "joint_x": data.joint("joint_x"),
                "joint_y": data.joint("joint_y")
            }
    ```

    WorldObjectBuilderのextract()の返り値はWorldBuilder::build()の返り値として取得できます．

    ```
    world, human_joints = WorldBuilder()
        .add_obj(
            Human()
        )
        .build()
    print(human_joints) // > {"human_factory":{"joint_x":<_MjDataJointViews>, "joint_y":<_MjDataJointViews>}}
    ```
    """

    def __init__(self, name: str):
        """
        コンストラクタ。

        Args:
            name (str): ビルダーの名前。WorldBuilderのbuildメソッドで返されるオブジェクトの辞書に登録される。
        """
        self.builder_name = name

    @abc.abstractmethod
    def gen_body(self) -> Body | None:
        """
        このオブジェクトを構成するBodyのXML要素を生成する抽象メソッド。

        Returns:
            Body | None: 生成されたBodyオブジェクト．Bodyを持たない場合はNoneを返す。
        """
        raise NotImplemented

    @abc.abstractmethod
    def gen_act(self) -> Actuator | None:
        """
        このオブジェクトに関わるActuatorのXML要素を生成する抽象メソッド。

        Returns:
            Actuator | None: 生成されたActuatorオブジェクト．Actuatorを持たない場合はNoneを返す。
        """
        raise NotImplemented

    @abc.abstractmethod
    def gen_sen(self) -> Sensor | None:
        """
        このオブジェクトに関わるSensorのXML要素を生成する抽象メソッド。

        Returns:
            Sensor | None: 生成されたSensorオブジェクト．Sensorを持たない場合はNoneを返す。
        """
        raise NotImplemented

    def _gen_all(self) -> tuple[Body | None, Actuator | None, Sensor | None]:
        return self.gen_body(), self.gen_act(), self.gen_sen()

    @abc.abstractmethod
    def extract(self, data: mujoco.MjData, timer: WorldClock):
        """
        MjDataから必要なデータの参照を取り出すための抽象メソッド。

        ここではMjDataから必要な値の参照を取得しそれをユーザに返す。
        返すobjectに制限はないため，参照をタプル型や辞書型にまとめて返すことも，構造体に整理して入れて返すこともできる．

        Args:
            data (mujoco.MjData): MuJoCoのデータオブジェクト。

            timer: MuJoCoのシミュレーション時間が格納されているクラス

        Returns:
            object: 任意のobjectを返すことができる．不要であればNoneを返す。
        """
        raise NotImplemented
