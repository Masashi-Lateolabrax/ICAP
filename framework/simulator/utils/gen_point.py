import numpy as np

from ..const import Settings, ROBOT_SIZE, FOOD_SIZE


def random_point_avoiding_invalid_areas(
        left_upper_point: tuple[float, float],
        right_lower_point: tuple[float, float],
        size: float,
        invalid_area: list[np.ndarray],
        retry: int = -1,
        padding: float = 0
) -> np.ndarray | None:
    """
    Generates a random point within a specified rectangular area while avoiding designated invalid regions.

    **Constraints:**
    - `left_upper_point` must represent the upper-left corner of the rectangle, meaning:
        - `left_upper_point[0]` < `right_lower_point[0]` (left x-coordinate is less than right x-coordinate)
        - `left_upper_point[1]` > `right_lower_point[1]` (upper y-coordinate is greater than lower y-coordinate)

    Parameters
    ----------
    left_upper_point : tuple of float
        The (x, y) coordinates of the upper-left corner of the rectangular area.
    right_lower_point : tuple of float
        The (x, y) coordinates of the lower-right corner of the rectangular area.
    size : float
        The size of the object to be placed within the rectangular area.
    invalid_area : list of np.ndarray
        A list of NumPy arrays representing invalid regions. Each array should contain at least three elements:
        the first two elements are the (x, y) coordinates of the center of the invalid area, and the third
        element is the radius of the invalid area.
    retry : int, optional
        The number of attempts to generate a valid point. If set to -1, the function will retry indefinitely
        until a valid point is found. The default value is -1.
    padding : float, optional
        The padding around the rectangular area. The default value is 0.

    Returns
    -------
    np.ndarray or None
        A NumPy array containing the (x, y) coordinates of the generated point that does not lie within any
        invalid area. If a valid point is not found within the specified number of retries, the function returns
        `None`.
    """

    pos = np.zeros(2)
    while retry < 0 or retry > 0:
        pos[0] = np.random.uniform(
            left_upper_point[0] + size + padding, right_lower_point[0] - size - padding
        )
        pos[1] = np.random.uniform(
            right_lower_point[1] + size + padding, left_upper_point[1] - size - padding
        )

        if np.any([np.linalg.norm(area[:2] - pos) <= area[2] + size for area in invalid_area]):
            if retry > 0:
                retry -= 1
            continue
        break

    return pos if retry != 0 else None


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
        (settings.Simulation.WORLD_WIDTH * -0.5, settings.Simulation.WORLD_HEIGHT * 0.5),
        (settings.Simulation.WORLD_WIDTH * 0.5, settings.Simulation.WORLD_HEIGHT * -0.5),
        ROBOT_SIZE,
        invalid_area,
        padding=ROBOT_SIZE
    )
    angle = np.random.uniform(0, 360)

    invalid_area.append(
        np.array([pos[0], pos[1], ROBOT_SIZE])
    )

    return pos[0], pos[1], angle


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
