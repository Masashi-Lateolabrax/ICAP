from typing import Callable

import mujoco
import numpy as np

_BRIGHTNESS_COEFFICIENT = np.array([0.299, 0.587, 0.114], dtype=np.float32)


class DistanceMeasure2Buf:
    def __init__(self, n):
        self.n = n

        self.intersected_id = np.zeros((n,), dtype=np.int32)
        self.intersected_dist = np.zeros((n,), dtype=np.float64)
        self.calc_sight_buf = np.zeros((1, n, 3), dtype=np.float32)
        self.buf1 = np.zeros((3, 1), dtype=np.float64)
        self.buf2 = np.zeros((3, 1), dtype=np.float64)
        self.vecs = np.zeros((n, 3))

        self.unit_quat = np.zeros((4,))
        mujoco.mju_axisAngle2Quat(self.unit_quat, [0, 0, 1], -mujoco.mjPI * 2 / n)


class DistanceMeasure2:
    def __init__(self, n: int, buf: DistanceMeasure2Buf | None = None):
        """LiDAR Sensor.

        LiDAR stands for 'Light Detection and Ranging', which measures distances to surrounding objects using lasers.

        Parameters
        ----------
        n: Integer
            The number of laser to measure the distances.

        buf: DistanceMeasureBuf, optional
            External buffer. If you set a buffer, the methods use the buffer, otherwise a buffer is automatically
            allocated internally.
        """

        self._n = n

        if buf is None:
            self._buf = DistanceMeasure2Buf(self._n)
        elif buf.n != self._n:
            import warnings
            warnings.warn(
                "A DistanceMeasure creates a buffer despite specifying the buffer to use, because the sizes did not match."
            )
            self._buf = DistanceMeasure2Buf(self._n)
        else:
            self._buf = buf

    def get_buf(self):
        return self._buf

    def measure(
            self,
            m: mujoco.MjModel, d: mujoco.MjData,
            position: np.ndarray, direction: np.ndarray,
            bodyexclude: int | None = None, cutoff: float = 100
    ):
        """
        Measuring distances to surrounding objects.

        Parameters
        ----------
        m: MjModel

        d: MjData

        position: (3,1) shape float64 ndarray
            The starting point position for LiDAR sensing.

        direction: (3,1) shape float64 ndarray
            A reference direction is a unit vector that specifies the starting direction from which LiDAR
            measurements begin.

        bodyexclude: int, optional
            The ID of a body excluded from LiDAR sensing.

        cutoff: float, default 100
            The max radius of LiDAR sensing.

        Returns
        -------
        intersected_id: n-dim int32 ndarray
            Object IDs which are intersected by laser.

        intersected_dist: n-dim float64 ndarray
            Distances to the intersected objects by laser.
        """

        intersected_id = self._buf.intersected_id
        intersected_dist = self._buf.intersected_dist
        buf1 = self._buf.buf1
        buf2 = self._buf.buf2
        vecs = self._buf.vecs
        unit_quat = self._buf.unit_quat

        center_point = np.copy(position)
        center_point[2] *= 0.5

        vecs.fill(0.0)
        np.copyto(buf1, -direction)
        for i in range(self._n):
            if i % 2 == 0:
                a = buf1
                b = buf2
            else:
                a = buf2
                b = buf1
            np.copyto(vecs[i], a[:, 0])
            mujoco.mju_rotVecQuat(b, a, unit_quat)

        mujoco.mj_multiRay(
            m, d,
            pnt=center_point,
            vec=vecs.reshape((self._n * 3,)),
            geomgroup=None,
            flg_static=1,
            bodyexclude=bodyexclude,
            geomid=intersected_id,
            dist=intersected_dist,
            nray=self._n,
            cutoff=cutoff,
        )

        return intersected_id, intersected_dist

    def measure_with_color_img(
            self,
            m: mujoco.MjModel, d: mujoco.MjData,
            position: np.ndarray, direction: np.ndarray,
            color_map: Callable[[str], np.ndarray],
            gain: Callable[[float], float],
            bodyexclude: int | None = None,
            cutoff: float = 100,
    ):
        """
        Measuring distances to surrounding objects.

        Parameters
        ----------
        m: MjModel

        d: MjData

        position: (3,1) shape float64 ndarray
            The starting point position for LiDAR sensing.

        direction: (3,1) shape float64 ndarray
            A reference direction is a unit vector that specifies the starting direction from which LiDAR
            measurements begin.

        color_map: functional(name:str) -> 3-dim float32 ndarray

        gain: functional(distance:float) -> float

        bodyexclude: int, optional
            The ID of a body excluded from LiDAR sensing.

        cutoff: float, optional, default 100
            The max radius of LiDAR sensing.

        Returns
        -------
            image: (1,n,3) shape ndarray
        """

        ids, dists = self.measure(m, d, position, direction, bodyexclude, cutoff)

        calc_sight_buf = self._buf.calc_sight_buf
        calc_sight_buf.fill(0)
        for j, (id_, dist) in enumerate(zip(ids, dists)):
            if id_ < 0:
                continue
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, id_)
            calc_sight_buf[0, j, :] = color_map(name) * gain(dist)

        return calc_sight_buf

    def measure_with_brightness_img(
            self,
            m: mujoco.MjModel, d: mujoco.MjData,
            position: np.ndarray, direction: np.ndarray,
            color_map: Callable[[str], np.ndarray],
            gain: Callable[[float], float],
            bodyexclude: int | None = None,
            cutoff: float = 100
    ):
        """
        Measuring distances to surrounding objects.

        Parameters
        ----------
        m: MjModel

        d: MjData

        position: (3,1) shape float64 ndarray
            The starting point position for LiDAR sensing.

        direction: (3,1) shape float64 ndarray
            A reference direction is a unit vector that specifies the starting direction from which LiDAR
            measurements begin.

        color_map: functional(name:str) -> 3-dim float32 ndarray

        gain: functional(distance:float) -> float

        bodyexclude: int, optional
            The ID of a body excluded from LiDAR sensing.

        cutoff: float, optional, default 100
            The max radius of LiDAR sensing.

        Returns
        -------
        brightness_image: (1,n) shape ndarray
        """
        color_img = self.measure_with_color_img(m, d, position, direction, color_map, gain, bodyexclude, cutoff)
        return np.dot(color_img, _BRIGHTNESS_COEFFICIENT)


class DistanceMeasure:
    def __init__(self, n, color_map: dict[str, tuple[float, float, float]], gain=lambda d: 1 / (d * 0.05 + 1)):
        self.n = n
        self.color_map = color_map
        self.gain = gain

        self._intersected_id = np.zeros((n,), dtype=np.int32)
        self._intersected_dist = np.zeros((n,), dtype=np.float64)

        self._unit_quat = np.zeros((4,))
        mujoco.mju_axisAngle2Quat(self._unit_quat, [0, 0, 1], -mujoco.mjPI * 2 / n)

        self._buf1 = np.zeros((3, 1), dtype=np.float64)
        self._buf2 = np.zeros((3, 1), dtype=np.float64)
        self._vecs = np.zeros((n, 3))
        self._calc_sight_buf = np.zeros((1, n, 3), dtype=np.uint8)

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
            self,
            m: mujoco.MjModel, d: mujoco.MjData,
            bot_body_id: int, bot_body, bot_direction: np.ndarray,
    ):
        ids, dists = self.measure(m, d, bot_body_id, bot_body, bot_direction)

        self._calc_sight_buf.fill(0)
        color = np.zeros((3,), dtype=np.float32)
        for j, (id_, dist) in enumerate(zip(ids, dists)):
            if id_ < 0:
                continue

            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, id_)
            color.fill(0.0)
            for key, c in self.color_map.items():
                if key in name:
                    color[:] = c
                    break

            self._calc_sight_buf[0, j, :] = color * self.gain(dist)
        return self._calc_sight_buf
