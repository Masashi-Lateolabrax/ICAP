import colorsys
import numpy
import scipy
import mujoco
from studyLib import wrap_mjc


class PheromoneField:
    def __init__(
            self,
            px: float, py: float,
            size: float, ds: float,
            nx: int, ny: int,
            sv: float,
            evaporate: float, diffusion: float, decrease: float,
            model: wrap_mjc.WrappedModel = None, z: float = -1,
    ):
        # セルの数
        self._nx = nx
        self._ny = ny

        # MuJoCo上に表示する座標と一つのセルの大きさ．
        self._px = px
        self._py = py
        self._size = size

        self._liquid = numpy.zeros((ny, nx))
        self._gas = numpy.zeros((ny, nx))

        self._evaporate = evaporate
        self._sv = sv

        self._diffusion = diffusion
        self._decrease = decrease

        self._dif_gas = numpy.zeros((ny, nx))
        self._dif_liquid = numpy.zeros((ny, nx))

        self._convolve = numpy.array([
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0],
        ]) * self._diffusion / (ds * ds)

        self._panels = []
        if model is not None:
            for y in range(0, self._ny):
                for x in range(0, self._nx):
                    pos = [(x - self._nx / 2) * self._size + self._px, (y - self._ny / 2) * self._size + self._py, z]
                    deco_geom = model.add_deco_geom(mujoco.mjtGeom.mjGEOM_PLANE)
                    deco_geom.set_pos(pos)
                    deco_geom.set_size([self._size * 0.5, self._size * 0.5, 0.05])
                    deco_geom.set_quat([0, 0, 1], 0)
                    self._panels.append(deco_geom)

    def _pos_to_index(self, x: float, y: float) -> (float, float):
        """
        MuJoCo上の座標から対応する要素のインデックスを返す．
        :param x: MuJoCo上のx座標
        :param y: MuJoCo上のy座標
        :return: 対応するself._gasのインデックス
        """
        ix = (x - self._px) / self._size + self._nx / 2
        iy = (y - self._py) / self._size + self._ny / 2
        return ix, iy

    def _calc_dico(self, x: float, y: float):
        """
        MuJoCo上の座標の付近に対応するself._gas上の要素のインデックスとその要素の重みを返す．
        :param x: MuJoCo上のX座標
        :param y: MuJoCo上のY座標
        :return: 付近の4つの要素のインデックスと重み．type: (インデックスX, インデックスY, 重み)．
        """
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

    def _add_liquid(self, ix: int, iy: int, value: float) -> None:
        """
        液体の量を表す指定した要素に値を加算する．
        :param ix: 要素のインデックス．
        :param iy: 要素のインデックス．
        :param value: 加算する値．
        :return: None
        """
        if 0 <= ix < self._nx and 0 <= iy < self._ny:
            self._liquid[iy, ix] += value

    def add_liquid(self, x: float, y: float, value: float) -> None:
        """
        液体の量を表す指定した座標の付近の複数の要素に重みづけされた指定した量を加える．
        :param x: MuJoCo上のX座標．
        :param y: MuJoCo上のY座標．
        :param value: 加算する値．
        :return:
        """
        dico = self._calc_dico(x, y)
        for x, y, e in dico:
            self._add_liquid(x, y, value * e)

    def _get_gas(self, ix: int, iy: int) -> float:
        if 0 <= ix < self._nx and 0 <= iy < self._ny:
            return self._gas[iy, ix]
        return 0.0

    def get_gas(self, x: float, y: float) -> float:
        dico = self._calc_dico(x, y)
        a = 0
        for x, y, e in dico:
            a += e * self._get_gas(x, y)
        return a

    def _get_deco_geom(self, ix: int, iy: int):
        if len(self._panels) == 0:
            return None
        return self._panels[self._nx * iy + ix]

    def _set_deco_geom_color(self, ix: int, iy: int, color: (float, float, float, float)):
        plane = self._get_deco_geom(ix, iy)
        plane.set_rgba(color)

    def update_cells(self, dt: float = 0.03333):
        eva = numpy.minimum(self._liquid, (self._sv - self._gas) * self._evaporate)
        self._dif_liquid -= eva

        dif = scipy.signal.convolve2d(self._gas, self._convolve, "same")
        dec = self._gas * self._decrease
        self._dif_gas += numpy.maximum(-self._gas, eva + dif - dec)

        self._liquid += self._dif_liquid * dt
        self._gas += self._dif_gas * dt

        self._dif_liquid.fill(0)
        self._dif_gas.fill(0)

    def update_panels(self, func=lambda x: x):
        if len(self._panels) != 0:
            for p, g in zip(self._panels, numpy.clip(self._gas / 10.0, 0.0, 1.0).ravel()):
                v = func(g)
                color = colorsys.hsv_to_rgb(0.66 * (1.0 - v), 1.0, 1.0)
                p.set_rgba((color[0], color[1], color[2], 0.7))
