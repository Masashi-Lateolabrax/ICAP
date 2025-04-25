import numpy as np

from ..const import Settings
from ..objects.robot import BrainInterface, RobotBuilder
from ..objects.food import FoodBuilder

from .gen_point import rand_robot_pos, rand_food_pos


def create_robot_builders(settings: Settings, id_, brain: BrainInterface, invalid_area: list[np.ndarray] = None):
    """
    ランダムな座標が設定されたロボットビルダーを作成する関数。

    指定されたIDと無効エリアを基に、ロボットのビルダーを作成します。
    また，作成されたロボットビルダーが占有するエリアを無効エリアに追加します。

    Args:
        settings (Settings): 設定

        id_ (int): ロボットのID

        brain (BrainInterface): ロボットの脳

        invalid_area (list[np.ndarray]): 無効エリアのリスト. ndarrayは3要素のリストであり、
            そのうちの2要素は座標であり、残りの1要素はサイズです。
            副作用があり実行後に無効エリアが更新されます。

    Returns:
        RobotBuilder: ロボットのビルダー
    """
    pos_and_angle = rand_robot_pos(settings, invalid_area)
    builder = RobotBuilder(
        settings,
        id_,
        brain,
        pos_and_angle
    )
    return builder


def create_food_builders(settings: Settings, id_, invalid_area: list[np.ndarray] = None):
    """
    ランダムな座標が設定されたフードビルダーを作成する関数。

    指定されたIDと無効エリアを基に、フードのビルダーを作成します。
    また，作成されたフードビルダーが占有するエリアを無効エリアに追加します。

    Args:
        settings (Settings): 設定

        id_ (int): フードのID

        invalid_area (list[np.ndarray]): 無効エリアのリスト. ndarrayは3要素のリストであり、
            そのうちの2要素は座標であり、残りの1要素はサイズです。
            副作用があり実行後に無効エリアが更新されます。

    Returns:
        FoodBuilder: フードのビルダー
    """
    pos = rand_food_pos(settings, invalid_area)
    builder = FoodBuilder(id_, pos)
    return builder
