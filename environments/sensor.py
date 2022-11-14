import numpy


class OmniSensor:
    def __init__(self, center, rot_mat, one_third_point: float, eps: float = 1.0):
        self._one_third_point = 1.0 / one_third_point
        self._center = center
        self._inv_rot_mat = numpy.linalg.inv(rot_mat)
        self._eps = 1.0 / eps
        self.value = numpy.zeros(2)

    def reset(self):
        self.value[:] = 0

    def sense(self, pos):
        ref_pos = numpy.dot(self._inv_rot_mat, pos - self._center)[:2]
        dist = numpy.linalg.norm(ref_pos, ord=2)
        if dist != 0:
            self.value += ref_pos * self._eps / (2 * self._one_third_point * dist * dist + dist)
