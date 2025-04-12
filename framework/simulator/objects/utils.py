import numpy as np

from ...settings import Settings
from ..utils import random_point_avoiding_invalid_areas

from .robot import ROBOT_SIZE, RobotBuilder, BrainInterface
from framework.simulator.objects.food.food_builder import FOOD_SIZE, FoodBuilder


def rand_robot_pos(settings: Settings, invalid_area: list[np.ndarray] = None):
    """
    ロボットのランダムな位置を生成する関数。

    Args:
        settings (Settings): 設定

        invalid_area (list[np.ndarray]): 無効エリアのリスト. ndarrayは3要素のリストであり、
            そのうちの2要素は座標であり、残りの1要素はサイズです。

    Returns:
        tuple[float, float, float]: ロボットの位置と角度
    """
    pos = random_point_avoiding_invalid_areas(
        (settings.Simulation.WORLD_WIDTH * -0.5, settings.Simulation.WORLD_WIDTH * 0.5),
        (settings.Simulation.WORLD_WIDTH * 0.5, settings.Simulation.WORLD_WIDTH * -0.5),
        ROBOT_SIZE,
        invalid_area,
        padding=ROBOT_SIZE
    )
    angle = np.random.uniform(0, 360)

    invalid_area.append(
        np.array([pos[0], pos[1], ROBOT_SIZE])
    )

    return pos[0], pos[1], angle


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


def rand_food_pos(settings: Settings, invalid_area: list[np.ndarray] = None):
    """
    フードのランダムな位置を生成する関数。

    Args:
        settings (Settings): 設定

        invalid_area (list[np.ndarray]): 無効エリアのリスト. ndarrayは3要素のリストであり、
            そのうちの2要素は座標であり、残りの1要素はサイズです。

    Returns:
        tuple[float, float]: フードの位置
    """
    pos = random_point_avoiding_invalid_areas(
        (settings.Simulation.WORLD_WIDTH * -0.5, settings.Simulation.WORLD_HEIGHT * 0.5),
        (settings.Simulation.WORLD_WIDTH * 0.5, settings.Simulation.WORLD_HEIGHT * -0.5),
        FOOD_SIZE,
        invalid_area,
        padding=FOOD_SIZE
    )
    invalid_area.append(
        np.array([pos[0], pos[1], FOOD_SIZE])
    )
    return pos


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
