import numpy
import scipy


def _calc_dico(w: int, h: int, xi: float, yi: float):
    xc = int(xi + 0.5)
    yc = int(yi + 0.5)
    x1 = xc - 0.5
    y1 = yc - 0.5

    x = xi - x1
    y = yi - y1

    e1 = (1 - x) * (1 - y)
    e2 = x * (1 - y)
    e3 = (1 - x) * y
    e4 = x * y

    in_x1 = 0 <= xc - 1 < w
    in_x2 = 0 <= xc < w
    in_y1 = 0 <= yc - 1 < h
    in_y2 = 0 <= yc < h

    res = []
    if in_x1:
        if in_y1:
            res.append((xc - 1, yc - 1, e1))
        if in_y2:
            res.append((xc - 1, yc, e3))
    if in_x2:
        if in_y1:
            res.append((xc, yc - 1, e2))
        if in_y2:
            res.append((xc, yc, e4))
    return res


class PheromoneField:
    def __init__(
            self,
            nx: int, ny: int, d: float,
            sv: float, evaporate: float, diffusion: float, decrease: float,
            dt: float, iteration: int = 1
    ):
        # セルの数
        self._nx = nx
        self._ny = ny

        self.dt = dt
        self.iteration = iteration

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
        for x, y, e in _calc_dico(self._nx, self._ny, xi, yi):
            self._liquid[int(x), int(y)] += e * value

    def get_gas(self, xi: float, yi: float) -> float:
        res = 0.0
        for x, y, e in _calc_dico(self._nx, self._ny, xi, yi):
            res += e * self._gas[int(x), int(y)]
        return res

    def update(self):
        dt = self.dt / self.iteration
        for _ in range(self.iteration):
            dif_liquid = numpy.minimum(self._liquid, (self._sv - self._gas) * self._eva) * dt
            dif_gas = dif_liquid + (scipy.signal.convolve2d(self._gas, self._c, "same") - self._gas * self._dec) * dt
            self._liquid -= dif_liquid
            self._gas += dif_gas
