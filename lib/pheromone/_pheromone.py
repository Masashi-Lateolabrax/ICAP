import numpy
import scipy


def _calc_dico(xi: float, yi: float):
    x2 = int(xi + 0.5)
    y2 = int(yi + 0.5)
    x1 = x2 - 1
    y1 = y2 - 1

    x = xi - x1
    y = yi - y1

    e1 = (1 - x) * (1 - y)
    e2 = x * (1 - y)
    e3 = (1 - x) * y
    e4 = x * y

    return (x1, y1, e1), (x2, y1, e2), (x1, y2, e3), (x2, y2, e4)


class PheromoneField:
    def __init__(
            self,
            nx: int, ny: int, d: float,
            sv: float, evaporate: float, diffusion: float, decrease: float,
    ):
        # セルの数
        self._nx = nx
        self._ny = ny

        self._liquid = numpy.zeros((nx, ny))
        self._gas = numpy.zeros((nx, ny))

        self._eva = evaporate  # 蒸発速度
        self._sv = sv  # 飽和蒸気量
        self._diffusion = diffusion  # 拡散速度
        self._dec = decrease  # 分解速度

        self._c = numpy.array([
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0],
        ]) * self._diffusion / (d * d)

    def add_liquid(self, xi: float, yi: float, value: float):
        for x, y, e in _calc_dico(xi, yi):
            if x < 0 or y < 0 or self._nx <= x or self._ny <= y:
                continue
            self._liquid[x, y] += e * value

    def get_gas(self, xi: float, yi: float) -> float:
        res = 0.0
        for x, y, e in _calc_dico(xi, yi):
            if x < 0 or y < 0 or self._nx <= x or self._ny <= y:
                continue
            res += e * self._gas[x, y]
        return res

    def update(self, dt: float):
        dif_liquid = numpy.minimum(self._liquid, (self._sv - self._gas) * self._eva) * dt
        dif_gas = dif_liquid + (scipy.signal.convolve2d(self._gas, self._c, "same") - self._gas * self._dec) * dt
        self._liquid -= dif_liquid
        self._gas += dif_gas
