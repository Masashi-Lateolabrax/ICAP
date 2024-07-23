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


class PheromoneField2:
    def __init__(
            self,
            nx: int, ny: int, d: float,
            sv: float, evaporate: float, diffusion: float, decrease: float,
    ):
        # セルの数
        self._nx = nx
        self._ny = ny

        self._liquid = numpy.zeros((nx, ny))
        self._gas_master = numpy.zeros((nx + 2, ny + 2))
        self._gas = self._gas_master[1:-1, 1:-1]

        self._eva = evaporate  # 蒸発速度
        self._sv = sv  # 飽和蒸気量
        self._diffusion = diffusion  # 拡散速度
        self._dec = decrease  # 分解速度

        self._c = numpy.array([
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0],
        ]) * self._diffusion / (d * d)

    def get_field_size(self) -> tuple[int, int]:
        return self._nx, self._ny

    def get_sv(self):
        return self._sv

    def get_cell(self, xi: int, yi: int):
        return self._liquid[xi:xi + 1, yi:yi + 1], self._gas[xi:xi + 1, yi:yi + 1]

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
        for x, y, e in _calc_dico(self._nx + 2, self._ny + 2, xi + 1, yi + 1):
            res += e * self._gas_master[int(x), int(y)]
        return res

    def update(self, dt: float, iteration: int = 1):
        dt = dt / iteration
        for _ in range(iteration):
            dif_liquid = numpy.minimum(self._liquid, (self._sv - self._gas) * self._eva) * dt

            # padding
            self._gas_master[0, :] = self._gas_master[1, :]
            self._gas_master[-1, :] = self._gas_master[-2, :]
            self._gas_master[:, 0] = self._gas_master[:, 1]
            self._gas_master[:, -1] = self._gas_master[:, -2]

            convolved = scipy.signal.convolve2d(self._gas_master, self._c, mode="valid")

            dif_gas = dif_liquid + (convolved - self._gas * self._dec) * dt

            self._liquid -= dif_liquid
            self._gas += dif_gas


class PheromoneField:
    def __init__(
            self,
            nx: int, ny: int, d: float,
            sv: float, evaporate: float, diffusion: float, decrease: float,
            dt: float, iteration: int = 1
    ):
        self._pheromone_field = PheromoneField2(nx, ny, d, sv, evaporate, diffusion, decrease)

        self.dt = dt
        self.iteration = iteration

    def get_field_size(self) -> tuple[int, int]:
        return self._pheromone_field.get_field_size()

    def get_sv(self):
        return self._pheromone_field.get_sv()

    def get_cell(self, xi: int, yi: int):
        return self._pheromone_field.get_cell(xi, yi)

    def get_all_liquid(self):
        return self._pheromone_field.get_all_gas()

    def set_liquid(self, xi: int, yi: int, value: float):
        self._pheromone_field.set_liquid(xi, yi, value)

    def add_liquid(self, xi: float, yi: float, value: float):
        self._pheromone_field.add_liquid(xi, yi, value)

    def get_all_gas(self):
        return self._pheromone_field.get_all_gas()

    def get_gas(self, xi: float, yi: float) -> float:
        return self._pheromone_field.get_gas(xi, yi)

    def update(self):
        self._pheromone_field.update(self.dt, self.iteration)
