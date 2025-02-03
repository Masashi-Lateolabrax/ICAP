import mujoco

from mujoco_xml_generator.utils import DummyGeom


class World:
    """
    シミュレーションの環境を管理するクラス。

    # Functions
    - シミュレーションの制御
    - 低レベルな値やオブジェクトの管理

    # Note
    - 直接インスタンス化はできない。WorldBuilderを使用してインスタンスを作成する。

    # Details
    このクラスはMjModelとMjDataを保持し，シミュレーションの制御を行う。そのためユーザは低レベルな値（MuJoCoが提供する値）にアクセスするときはこのクラスを介して行う。

    シミュレータ上の物体はWorldObjectBuilderを介して作成する．WorldObjectBuilderを継承したクラスで物体の設計を行う．
    そしてのクラスをWorldBuilderに渡すことでWorldの生成時に物体をシミュレータ上に追加できる．

    Worldの生成時にはWorldObjectBuilderのextractメソッドが呼ばれ物体のデータの取得が行われる．
    またその結果はWorldBuilderのbuildメソッドの返り値として返される．
    """

    def __init__(self):
        raise RuntimeError("Use WorldBuilder to create World instances")

    @classmethod
    def _create(cls, model: mujoco.MjModel):
        instance = cls.__new__(cls)
        instance.__init_instance(model)
        return instance

    def __init_instance(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(self.model)

    def calc_step(self):
        """
        シミュレーションを1ステップ進める。
        """
        mujoco.mj_step(self.model, self.data)

    def get_dummy_geoms(self) -> list[DummyGeom]:
        """
        描写用のDummyGeomを取得する。

        Returns:
            list[DummyGeom]: 描写用のDummyGeomのリスト。
        """
        return []
