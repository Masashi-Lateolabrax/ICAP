import mujoco
import numpy as np


class DistanceMeasure:
    def __init__(self, n):
        self.n = n
        self._intersected_id = np.zeros((n,), dtype=np.int32)
        self._intersected_dist = np.zeros((n,), dtype=np.float64)

        unit_quat = np.zeros((4,))
        mujoco.mju_axisAngle2Quat(unit_quat, [0, 0, 1], -mujoco.mjPI * 2 / n)

        buf1 = np.array([0, -1, 0], dtype=np.float64)
        buf2 = np.zeros((3,), dtype=np.float64)
        vecs = np.zeros((n, 3))
        for i in range(n):
            if i % 2 == 0:
                a = buf1
                b = buf2
            else:
                a = buf2
                b = buf1
            np.copyto(vecs[i], a)
            mujoco.mju_rotVecQuat(b, a, unit_quat)
        self.vecs = vecs.reshape((n * 3,)).copy()

    def measure(self, m: mujoco.MjModel, d: mujoco.MjData, bot_body_id: int):
        bot_body = d.body(bot_body_id)
        center_point = bot_body.xpos
        center_point[2] *= 0.5

        mujoco.mj_multiRay(
            m, d,
            pnt=center_point,
            vec=self.vecs,
            geomgroup=None,
            flg_static=1,
            bodyexclude=bot_body_id,
            geomid=self._intersected_id,
            dist=self._intersected_dist,
            nray=self.n,
            cutoff=100,
        )

        return self._intersected_id, self._intersected_dist
