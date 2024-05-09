import mujoco
import numpy as np


class DistanceMeasure:
    def __init__(self, n):
        self.n = n
        self._intersected_id = np.zeros((n,), dtype=np.int32)
        self._intersected_dist = np.zeros((n,), dtype=np.float64)

        self._unit_quat = np.zeros((4,))
        mujoco.mju_axisAngle2Quat(self._unit_quat, [0, 0, 1], -mujoco.mjPI * 2 / n)

        self._buf1 = np.zeros((3, 1), dtype=np.float64)
        self._buf2 = np.zeros((3, 1), dtype=np.float64)
        self._vecs = np.zeros((n, 3))
        self._calc_sight_buf = np.zeros((1, n + 1, 3), dtype=np.uint8)

    def measure(self, m: mujoco.MjModel, d: mujoco.MjData, bot_body_id: int, bot_body, bot_direction: np.ndarray):
        center_point = np.copy(bot_body.xpos)
        center_point[2] *= 0.5

        np.copyto(self._buf1, -bot_direction)
        self._vecs.fill(0.0)
        for i in range(self.n):
            if i % 2 == 0:
                a = self._buf1
                b = self._buf2
            else:
                a = self._buf2
                b = self._buf1
            np.copyto(self._vecs[i], a[:, 0])
            mujoco.mju_rotVecQuat(b, a, self._unit_quat)

        mujoco.mj_multiRay(
            m, d,
            pnt=center_point,
            vec=self._vecs.reshape((self.n * 3,)),
            geomgroup=None,
            flg_static=1,
            bodyexclude=bot_body_id,
            geomid=self._intersected_id,
            dist=self._intersected_dist,
            nray=self.n,
            cutoff=100,
        )

        return self._intersected_id, self._intersected_dist

    def measure_with_img(
            self, m: mujoco.MjModel, d: mujoco.MjData, bot_body_id: int, bot_body, bot_direction: np.ndarray
    ):
        ids, dists = self.measure(m, d, bot_body_id, bot_body, bot_direction)

        self._calc_sight_buf.fill(0)
        color = np.zeros((3,), dtype=np.float32)
        for j, (id_, dist) in enumerate(zip(ids, dists)):
            if id_ < 0:
                continue

            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, id_)
            if "bot" in name:
                color[:] = [255, 255, 0]
            elif "goal" in name:
                color[:] = [0, 255, 0]

            self._calc_sight_buf[0, j, :] = color / ((dist * 0.05 + 1) ** 2)

        self._calc_sight_buf[0, -1, :] = self._calc_sight_buf[0, 0, :]
        return self._calc_sight_buf
