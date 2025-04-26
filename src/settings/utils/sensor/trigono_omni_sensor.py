import numpy as np


def trigono_omni_sensor(center, direction, targets, f=lambda d: d) -> (float, float):
    direction = direction[0:2]
    orthogonal_direction = np.array([direction[1], -direction[0]])
    horizontal = 0
    vertical = 0
    n = 0
    for t in targets:
        n += 1
        sub = (t - center)[0:2]
        d = np.linalg.norm(sub)
        sub /= d
        horizontal += np.dot(orthogonal_direction, sub) / f(d)
        vertical += np.dot(direction, sub) / f(d)

    if n == 0:
        return 0, 0

    horizontal /= n
    vertical /= n
    return horizontal, vertical
