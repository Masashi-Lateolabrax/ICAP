import mujoco
import numpy as np


def axisangle_to_quat(axis: np.ndarray | tuple[float, float, float], angle: float, res: np.ndarray = None):
    if res is None:
        res = np.zeros(4)
    if isinstance(axis, tuple):
        axis = np.array(axis)

    mujoco.mju_axisAngle2Quat(res, axis, angle)
    return res
