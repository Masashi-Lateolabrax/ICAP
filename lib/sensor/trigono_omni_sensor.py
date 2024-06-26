import numpy as np


def trigono_omni_sensor(
        center: np.ndarray, direction: np.ndarray, targets: np.ndarray, f=lambda d: d
) -> (float, float):
    if center.shape != (2,):
        raise "Invalid argument. the shape of 'center' must be (2,)."
    elif direction.shape != (2,):
        raise "Invalid argument. the shape of 'direction' must be (2,)."
    elif targets.ndim != 2 or targets.shape[1] != 2:
        raise "Invalid argument. the shape of 'targets' must be (n,2)."

    if targets.shape[0] == 0:
        return 0, 0

    f = np.frompyfunc(f, 1, 1)

    direction = np.array([
        [direction[1], -direction[0]],
        [direction[0], direction[1]],
    ])

    sub = targets - center
    distance = np.linalg.norm(sub, axis=1)
    trigonal_components = np.dot(direction, sub.T / distance).T
    result = np.dot(
        f(distance),
        trigonal_components
    ) / targets.shape[0]

    return result[0], result[1]


def _test():
    center = np.array([0, 0])
    direction = np.array([3 / 5, 4 / 5])
    targets = np.array([
        [1, 1],
        [2, 3],
        [-1, 1]
    ])
    result = trigono_omni_sensor(center, direction, targets, lambda d: 1 / (0.001 * d + 1))
    print(result)


if __name__ == '__main__':
    _test()
