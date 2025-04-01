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

    hit = []
    if in_x1:
        if in_y1:
            hit.append((xc - 1, yc - 1, e1))
        if in_y2:
            hit.append((xc - 1, yc, e3))
    if in_x2:
        if in_y1:
            hit.append((xc, yc - 1, e2))
        if in_y2:
            hit.append((xc, yc, e4))

    return hit


class PheromoneCell:
    def __init__(self, liquid_cell: numpy.ndarray, gas_cell: numpy.ndarray):
        self._liquid = liquid_cell
        self._gas = gas_cell

    def set_gas(self, value):
        self._gas[0, 0] = value

    def set_liquid(self, value):
        self._liquid[0, 0] = value

    def get_gas(self):
        return self._gas[0, 0]

    def get_liquid(self):
        return self._liquid[0, 0]


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

        self._rk_buf = numpy.zeros((4, 2, nx, ny))
        self._rk_arg_buf = numpy.zeros((2, nx, ny))

    def get_field_size(self) -> tuple[int, int]:
        return self._nx, self._ny

    def get_sv(self):
        return self._sv

    def get_cell(self, xi: int, yi: int) -> PheromoneCell:
        cell = PheromoneCell(self._liquid[xi:xi + 1, yi:yi + 1], self._gas[xi:xi + 1, yi:yi + 1])
        return cell

    def get_all_liquid(self):
        return numpy.copy(self._liquid)

    def set_liquid(self, xi: int, yi: int, value: float):
        for x, y, e in _calc_dico(self._nx, self._ny, xi, yi):
            self._liquid[int(x), int(y)] = e * value

    def add_liquid(self, xi: float, yi: float, value: float):
        for x, y, e in _calc_dico(self._nx, self._ny, xi, yi):
            self._liquid[int(x), int(y)] += e * value

    def get_all_gas(self):
        return numpy.copy(self._gas)

    def get_gas(self, xi: float, yi: float) -> float:
        res = 0.0
        for x, y, e in _calc_dico(self._nx, self._ny, xi, yi):
            res += e * self._gas[int(x), int(y)]
        return res

    def _euler_update(self, dt: float):
        dif_liquid = numpy.minimum(self._liquid, (self._sv - self._gas) * self._eva) * dt

        convolved = scipy.signal.convolve2d(self._gas, self._c, mode="same", boundary="symm")
        dif_gas = dif_liquid + (convolved - self._gas * self._dec) * dt

        self._liquid -= dif_liquid
        self._gas += dif_gas

    def _rk_update(self, dt: float):
        def f(liq, g, res):
            res[0, :, :] = -numpy.minimum(liq, self._eva * (self._sv - g))
            lap = scipy.signal.convolve2d(g, self._c, mode="same", boundary="symm")
            res[1, :, :] = self._diffusion * lap - self._dec * g - res[0, :, :]

        # Updating
        f(self._liquid, self._gas, self._rk_buf[0])

        self._rk_arg_buf[0] = self._rk_buf[0, 0]
        self._rk_arg_buf[0] *= 0.5 * dt
        self._rk_arg_buf[0] += self._liquid
        self._rk_arg_buf[1] = self._rk_buf[0, 1]
        self._rk_arg_buf[1] *= 0.5 * dt
        self._rk_arg_buf[1] += self._gas
        f(self._rk_arg_buf[0], self._rk_arg_buf[1], self._rk_buf[1])

        self._rk_arg_buf[0] = self._rk_buf[1, 0]
        self._rk_arg_buf[0] *= 0.5 * dt
        self._rk_arg_buf[0] += self._liquid
        self._rk_arg_buf[1] = self._rk_buf[1, 1]
        self._rk_arg_buf[1] *= 0.5 * dt
        self._rk_arg_buf[1] += self._gas
        f(self._rk_arg_buf[0], self._rk_arg_buf[1], self._rk_buf[2])

        self._rk_arg_buf[0] = self._rk_buf[2, 0]
        self._rk_arg_buf[0] *= dt
        self._rk_arg_buf[0] += self._liquid
        self._rk_arg_buf[1] = self._rk_buf[2, 1]
        self._rk_arg_buf[1] *= dt
        self._rk_arg_buf[1] += self._gas
        f(self._rk_arg_buf[0], self._rk_arg_buf[1], self._rk_buf[3])

        dy = dt / 6 * (self._rk_buf[0] + 2 * self._rk_buf[1] + 2 * self._rk_buf[2] + self._rk_buf[3])
        self._liquid += dy[0]
        self._gas += dy[1]

    def update(self, dt: float, iteration: int = 1):
        dt = dt / iteration
        for _ in range(iteration):
            # self._euler_update(dt)
            self._rk_update(dt)
