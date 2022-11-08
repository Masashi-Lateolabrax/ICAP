import numpy
import scipy
import studyLib
import mujoco


class PheromoneField:
    def __init__(
            self,
            px: float, py: float,
            size: float, ds: float,
            nx: int, ny: int,
            evaporate: float, diffusion: float, decrease: float
    ):
        self.px = px
        self.py = py
        self.nx = nx
        self.ny = ny
        self.size = size
        self.ds = ds * ds
        self.gas = numpy.zeros(nx * ny).reshape(ny, nx)
        self.liquid = numpy.zeros(nx * ny).reshape(ny, nx)
        self.dif_gas = numpy.zeros(nx * ny).reshape(ny, nx)
        self.dif_liquid = numpy.zeros(nx * ny).reshape(ny, nx)
        self.evaporate = evaporate
        self.diffusion = diffusion
        self.decrease = decrease

    def _pos_to_index(self, x: float, y: float) -> (float, float):
        ix = (x - self.px) / self.size + self.nx / 2
        iy = (y - self.py) / self.size + self.ny / 2
        return ix, iy

    def _calc_dico(self, x: float, y: float):
        cx, cy = self._pos_to_index(x, y)
        x1 = int(cx)
        y1 = int(cy)
        x2 = x1 + 1
        y2 = y1 + 1
        x = cx - x1
        y = cy - y1
        e1 = (1 - x) * (1 - y)
        e2 = x * (1 - y)
        e3 = (1 - x) * y
        e4 = x * y
        return (x1, y1, e1), (x2, y1, e2), (x1, y2, e3), (x2, y2, e4)

    def _add_liquid(self, ix: int, iy: int, value: float):
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            self.dif_liquid[iy, ix] += value

    def add_liquid(self, x: float, y: float, value: float):
        dico = self._calc_dico(x, y)
        for x, y, e in dico:
            self._add_liquid(x, y, value * e)

    def _get_gas(self, ix: int, iy: int) -> float:
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            return self.gas[iy, ix]
        return 0.0

    def get_gas(self, x: float, y: float) -> float:
        dico = self._calc_dico(x, y)
        a = 0
        for x, y, e in dico:
            a += e * self._get_gas(x, y)
        return a

    def update(self):
        c = numpy.array([
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0],
        ]) * self.diffusion / self.ds
        eva = self.liquid * self.evaporate
        dif = scipy.signal.convolve2d(self.gas, c, "same")
        dec = self.gas * self.decrease
        self.dif_liquid -= eva
        self.dif_gas += eva + dif - dec
        self.liquid += self.dif_liquid
        self.gas += self.dif_gas
        self.dif_liquid *= 0
        self.dif_gas *= 0
        pass

    def render(self, model: studyLib.WrappedModel):
        for y in range(0, self.ny):
            for x in range(0, self.nx):
                pos = [(x - self.nx / 2) * self.size + self.px, (y - self.ny / 2) * self.size + self.py, -8]
                deco_geom = model.add_deco_geom(mujoco.mjtGeom.mjGEOM_PLANE)
                deco_geom.set_pos(pos)
                deco_geom.set_size([self.size * 0.5, self.size * 0.5, 0.05])
                deco_geom.set_quat([0, 0, 1], 0)

                value = self.gas[y, x]
                if value > 1.0:
                    value = 1.0
                elif value < 0.0:
                    value = 0.0
                deco_geom.set_rgba([value, 0.0, 1.0 - value, 0.5])
