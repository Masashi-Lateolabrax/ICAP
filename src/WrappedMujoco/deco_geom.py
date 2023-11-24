import mujoco
import numpy
import math


class DecoGeom(mujoco.MjvGeom):
    def __init__(self, geom_type: mujoco.mjtGeom):
        super().__init__()
        mujoco.mjv_initGeom(
            self, geom_type,
            numpy.ones(3), numpy.ones(3), numpy.eye(3, 3).ravel(),
            numpy.ones(4).astype(numpy.float32)
        )

    def copy_to(self, destination: mujoco.MjvGeom):
        mujoco.mjv_initGeom(
            destination,
            self.type,
            self.size,
            self.pos,
            self.mat.ravel(),
            self.rgba,
        )

    def get_type(self) -> int:
        return self.type

    def get_size(self) -> numpy.ndarray:
        return self.size.copy()

    def set_size(self, size):
        self.size = numpy.array(size).reshape((3,))

    def get_pos(self) -> numpy.ndarray:
        return self.pos.copy()

    def set_pos(self, pos):
        self.pos = numpy.array(pos).reshape((3,))

    def get_quat(self) -> numpy.ndarray:
        quat = numpy.zeros(4)
        mujoco.mju_mat2Quat(quat, self.mat.ravel())
        return quat

    def set_quat(self, axis, theta):
        quat = numpy.zeros(4)
        quat[0] = math.cos(theta * 0.5)
        quat[1] = axis[0] * math.sin(theta * 0.5)
        quat[2] = axis[1] * math.sin(theta * 0.5)
        quat[3] = axis[2] * math.sin(theta * 0.5)
        mat = numpy.zeros(9)
        mujoco.mju_quat2Mat(mat, quat)
        self.mat = mat.reshape(3, 3)

    def get_rgba(self) -> numpy.ndarray:
        return self.rgba.copy()

    def set_rgba(self, rgba):
        self.rgba = numpy.array(rgba).reshape((4,))
