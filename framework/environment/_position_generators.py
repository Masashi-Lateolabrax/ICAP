import numpy as np

from ..prelude import *


def check_collision(
        pos: np.ndarray,
        size: float,
        invalid_area: list[tuple[Position, float]],
) -> bool:
    return np.any([np.linalg.norm(np.array([area[0].x, area[0].y]) - pos) <= area[1] + size for area in invalid_area])


def random_point_avoiding_invalid_areas(
        left_upper_point: tuple[float, float],
        right_lower_point: tuple[float, float],
        size: float,
        invalid_area: list[tuple[Position, float]],
        retry: int = -1,
        padding: float = 0,
        rng: np.random.Generator = None
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
        The (x, y) coordinates of the upper-left corner at the rectangular area.
    right_lower_point : tuple of float
        The (x, y) coordinates of the lower-right corner at the rectangular area.
    size : float
        The size of the object to be placed within the rectangular area.
    invalid_area : list[tuple[Position, float]]
        A list of tuples representing invalid regions. Each tuple contains a Position object (with x, y coordinates)
        as the first element and a float radius as the second element.
    retry : int, optional
        The number of attempts to generate a valid point. If set to -1, the function will retry indefinitely
        until a valid point is found. The default value is -1.
    padding : float, optional
        The padding around the rectangular area. The default value is 0.
    rng : np.random.Generator, optional
        Random number generator to use. If None, uses the default numpy random generator.

    Returns
    -------
    np.ndarray or None
        A NumPy array containing the (x, y) coordinates of the generated point that does not lie within any
        invalid area. If a valid point is not found within the specified number of retries, the function returns
        `None`.
    """
    if rng is None:
        rng = np.random.default_rng()

    pos = np.zeros(2)
    while retry < 0 or retry > 0:
        pos[0] = rng.uniform(
            left_upper_point[0] + size + padding, right_lower_point[0] - size - padding
        )
        pos[1] = rng.uniform(
            right_lower_point[1] + size + padding, left_upper_point[1] - size - padding
        )

        if check_collision(pos, size, invalid_area):
            if retry > 0:
                retry -= 1
            continue
        break

    return pos if retry != 0 else None


def rand_robot_pos(
        world_width: float, world_height: float,
        robot_radius: float,
        invalid_area: list[tuple[Position, float]] = None
) -> RobotLocation:
    """
    ロボットのランダムな位置を生成する関数。

    Args:
        world_width (float): ワールドの幅
        world_height (float): ワールドの高さ
        robot_radius (float): ロボットの半径
        invalid_area (list[tuple[Position, float]]): 無効エリアのリスト. 各タプルはPosition（x, y座標）と
            半径（float）を含みます。

    Returns:
        RobotLocation: ロボットの位置と角度
    """
    pos = random_point_avoiding_invalid_areas(
        (world_width * -0.5, world_height * 0.5),
        (world_width * 0.5, world_height * -0.5),
        robot_radius,
        invalid_area if invalid_area is not None else [],
        padding=robot_radius
    )
    angle = np.random.uniform(0, 360)
    return RobotLocation(pos[0], pos[1], angle)


def rand_food_pos(
        world_width: float, world_height: float,
        food_radius: float,
        invalid_area: list[tuple[Position, float]] = None,
        rng: np.random.Generator = None
) -> Position:
    """
    フードのランダムな位置を生成する関数。

    Args:
        world_width (float): ワールドの幅
        world_height (float): ワールドの高さ
        food_radius (float): フードの半径
        invalid_area (list[tuple[Position, float]]): 無効エリアのリスト. 各タプルはPosition（x, y座標）と
            半径（float）を含みます。
        rng (np.random.Generator): 乱数生成器。Noneの場合はデフォルトの乱数生成器を使用。

    Returns:
        Position: フードの位置
    """
    pos = random_point_avoiding_invalid_areas(
        (world_width * -0.5, world_height * 0.5),
        (world_width * 0.5, world_height * -0.5),
        food_radius,
        invalid_area if invalid_area is not None else [],
        padding=food_radius,
        rng=rng
    )
    return Position(pos[0], pos[1])
