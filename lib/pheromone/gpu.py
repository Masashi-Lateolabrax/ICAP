import numpy
import cupy as cp
from cupyx.scipy.ndimage import convolve


def _calc_dico(w: int, h: int, xi: float, yi: float, out: numpy.ndarray) -> tuple[int, int]:
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

    out[0, 0] = e1 if 0 <= x1 < w and 0 <= y1 < h else 0.0
    out[1, 0] = e2 if 0 <= xc < w and 0 <= y1 < h else 0.0
    out[0, 1] = e3 if 0 <= x1 < w and 0 <= yc < h else 0.0
    out[1, 1] = e4 if 0 <= xc < w and 0 <= yc < h else 0.0

    return xc - 1, yc - 1


class PheromoneField:
    def __init__(
            self,
            width: int, height: int, d: float,
            sv: float, evaporate: float, diffusion: float, decrease: float,
            timestep: float, iteration: int = 1,
    ):
        cp.cuda.Device(0).use()

        # セルの数
        self.width = width
        self.height = height

        self.dt = timestep
        self.iteration = iteration

        self._dico_out = numpy.zeros((2, 2), dtype=numpy.float32)
        self.liquid = cp.zeros((width, height), dtype=cp.float32)
        self.gas = cp.zeros((width, height), dtype=cp.float32)

        self._eva = evaporate  # 蒸発速度
        self._sv = sv  # 飽和蒸気量
        self._diffusion = diffusion  # 拡散速度
        self._dec = decrease  # 分解速度

        self.kernel = cp.array([
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0],
        ], dtype=cp.float32) * diffusion / (d * d)

    def add_liquid(self, xi: float, yi: float, value: float):
        x, y = _calc_dico(self.width, self.height, xi, yi, self._dico_out)
        self.liquid[x:x + 2, y:y + 2] += cp.asarray(self._dico_out * value)

    def get_gas(self, xi: float, yi: float) -> float:
        res = 0.0
        x, y = _calc_dico(self.width, self.height, xi, yi, self._dico_out)
        res += numpy.sum(self._dico_out * cp.asnumpy(self.gas[x:x + 2, y:y + 2]))
        return res

    def update(self):
        dt = self.dt / self.iteration

        for _ in range(self.iteration):
            dif_liquid = cp.minimum(self.liquid, (self._sv - self.gas) * self._eva) * dt
            c = convolve(self.gas, self.kernel)
            dif_gas = dif_liquid + (c - self.gas * self._dec) * dt

            self.liquid -= dif_liquid
            self.gas += dif_gas
