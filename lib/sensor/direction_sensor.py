import numpy as np


def direction_sensor(
        center: np.ndarray, direction: np.ndarray, target: np.ndarray, r: float
) -> (float, float):
    if center.shape != (2,):
        raise "Invalid argument. the shape of 'center' must be (2,)."
    elif direction.shape != (2,):
        raise "Invalid argument. the shape of 'direction' must be (2,)."
    elif target.shape != (2,):
        raise "Invalid argument. the shape of 'target' must be (2,)."

    direction = np.array([
        [direction[1], -direction[0]],
        [direction[0], direction[1]],
    ])

    sub = target - center
    distance = np.linalg.norm(sub)
    if distance > r:
        k = 1 / distance
    else:
        k = 1 / r
    result = np.dot(direction, sub * k)

    return result[0], result[1]


def _test():
    center = np.array([0, 0])
    direction = np.array([3 / 5, 4 / 5])
    target = np.array([1, 1])
    result = direction_sensor(center, direction, target, 2)
    print(result)


if __name__ == '__main__':
    _test()
