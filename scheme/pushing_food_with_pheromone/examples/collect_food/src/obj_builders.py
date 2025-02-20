import numpy as np

from scheme.pushing_food_with_pheromone.lib.utilities import random_point_avoiding_invalid_areas
from scheme.pushing_food_with_pheromone.lib.objects.robot import RobotBuilder, BrainInterface, RobotInput, \
    BrainJudgement
from scheme.pushing_food_with_pheromone.lib.objects.food import FoodBuilder

from .prerulde import Settings


class _DummyBrain(BrainInterface):
    @staticmethod
    def get_dim() -> int:
        return 0

    def think(self, input_: RobotInput) -> BrainJudgement:
        return BrainJudgement.STOP


def create_robot_builders(id_, invalid_area: list[np.ndarray] = None):
    """
    ランダムな座標が設定されたロボットビルダーを作成する関数。

    指定されたIDと無効エリアを基に、ロボットのビルダーを作成します。
    また，作成されたロボットビルダーが占有するエリアを無効エリアに追加します。

    Args:
        id_ (int): ロボットのID

        invalid_area (list[np.ndarray]): 無効エリアのリスト. ndarrayは3要素のリストであり、
            そのうちの2要素は座標であり、残りの1要素はサイズです。
            副作用があり実行後に無効エリアが更新されます。

    Returns:
        RobotBuilder: ロボットのビルダー
    """
    size = Settings.ROBOT_SIZE
    pos = random_point_avoiding_invalid_areas(
        (Settings.WORLD_WIDTH * -0.5, Settings.WORLD_HEIGHT * 0.5),
        (Settings.WORLD_WIDTH * 0.5, Settings.WORLD_HEIGHT * -0.5),
        size,
        invalid_area,
        padding=Settings.WALL_PADDING
    )
    angle = np.random.uniform(0, 360)
    builder = RobotBuilder(
        id_,
        _DummyBrain(),
        (pos[0], pos[1], angle),
        size, Settings.ROBOT_WEIGHT, Settings.SENSOR_GAIN, Settings.SENSOR_OFFSET,
        Settings.ROBOT_MOVE_SPEED, Settings.ROBOT_TURN_SPEED, Settings.FOOD_NUM
    )
    invalid_area.append(
        np.array([pos[0], pos[1], size])
    )
    return builder


def create_food_builders(id_, invalid_area: list[np.ndarray] = None):
    """
    ランダムな座標が設定されたフードビルダーを作成する関数。

    指定されたIDと無効エリアを基に、フードのビルダーを作成します。
    また，作成されたフードビルダーが占有するエリアを無効エリアに追加します。

    Args:
        id_ (int): フードのID
        invalid_area (list[np.ndarray]): 無効エリアのリスト. ndarrayは3要素のリストであり、
            そのうちの2要素は座標であり、残りの1要素はサイズです。
            副作用があり実行後に無効エリアが更新されます。

    Returns:
        FoodBuilder: フードのビルダー
    """
    size = Settings.FOOD_SIZE
    pos = random_point_avoiding_invalid_areas(
        (Settings.WORLD_WIDTH * -0.5, Settings.WORLD_HEIGHT * 0.5),
        (Settings.WORLD_WIDTH * 0.5, Settings.WORLD_HEIGHT * -0.5),
        size,
        invalid_area,
        padding=Settings.WALL_PADDING
    )
    builder = FoodBuilder(id_, pos, size, Settings.FOOD_DENSITY, Settings.FOOD_FRICTIONLOSS)
    invalid_area.append(
        np.array([pos[0], pos[1], size])
    )
    return builder
