import numpy as np


def direction_sensor(center, direction, target, r) -> (float, float):
    direction = direction[0:2]
    orthogonal_direction = np.array([direction[1], -direction[0]])

    sub = (target - center)[0:2]
    distance = np.linalg.norm(sub)
    sub_e = sub / distance
    k = distance / r if distance < r else 1.0

    horizontal = np.dot(orthogonal_direction, sub_e) * k
    vertical = np.dot(direction, sub_e) * k

    return horizontal, vertical
