import numpy as np

rng = np.random.default_rng()


def robot_names(i: int):
    return {
        "body": f"bot{i}.body",
        "geom": f"bot{i}.geom",
        "joint_x": f"bot{i}.joint.slide_x",
        "joint_y": f"bot{i}.joint.slide_y",
        "joint_r": f"bot{i}.joint.hinge",
        "camera": f"bot{i}.camera",
        "act_x": f"bot{i}.act.horizontal",
        "act_y": f"bot{i}.act.vertical",
        "act_r": f"bot{i}.act.rotation",
        "center_site": f"bot{i}.site.center",
        "front_site": f"bot{i}.site.front",
        "velocimeter": f"bot{i}.sensor.vel",
    }


def random_point_avoiding_invalid_areas(
        left_upper_point: tuple[float, float],
        right_lower_point: tuple[float, float],
        invalid_area: list[np.ndarray],
        retry=-1
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
    invalid_area : list of np.ndarray
        A list of NumPy arrays representing invalid regions. Each array should contain at least three elements:
        the first two elements are the (x, y) coordinates of the center of the invalid area, and the third
        element is the radius of the invalid area.
    retry : int, optional
        The number of attempts to generate a valid point. If set to -1, the function will retry indefinitely
        until a valid point is found. The default value is -1.

    Returns
    -------
    np.ndarray or None
        A NumPy array containing the (x, y) coordinates of the generated point that does not lie within any
        invalid area. If a valid point is not found within the specified number of retries, the function returns
        `None`.
    """

    pos = np.zeros(2)
    while retry < 0 or retry > 0:
        pos[0] = np.random.uniform(left_upper_point[0], right_lower_point[0])
        pos[1] = np.random.uniform(right_lower_point[1], left_upper_point[1])

        if np.any([np.linalg.norm(area[:2] - pos) < area[2] for area in invalid_area]):
            if retry > 0:
                retry -= 1
            continue
        break

    return pos if retry != 0 else None
