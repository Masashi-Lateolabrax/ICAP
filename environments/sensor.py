import numpy


class OmniSensor:
    def __init__(self, center, rot_mat, one_third_point: float, min_value: float = 1.0):
        if min_value <= 0:
            raise ValueError(f"'min_value' must be grater than 0 but it is {min_value}.")
        if one_third_point <= 0:
            raise ValueError(f"'one_third_point' must be grater than 0 but it is {one_third_point}.")

        self._a = 2.0 / one_third_point
        self._min = min_value

        self._center = center
        self._inv_rot_mat = numpy.linalg.inv(rot_mat)

        self.value = numpy.zeros(2)

    def reset(self):
        self.value[:] = 0

    def sense(self, pos):
        ref_pos = numpy.dot(self._inv_rot_mat, pos - self._center)
        d = numpy.linalg.norm(ref_pos[:2], ord=2)
        if d != 0:
            ref_direction = ref_pos[:2] / d
            self.value += self._min / (self._a * d + 1.0) * ref_direction
        else:
            self.value += self._min
