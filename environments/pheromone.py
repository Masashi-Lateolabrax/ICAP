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

        # セルの大きさ
        self._ds = ds

        # MuJoCo上に表示する座標と一つのセルの大きさ．
        self._px = px
        self._py = py
        self._size = size

        self._liquid = numpy.zeros((ny, nx))
        self._gas = numpy.zeros((ny, nx))

        self._eva = evaporate  # 蒸発速度
        self._sv = sv  # 飽和蒸気量

        self._diffusion = diffusion  # 拡散速度
        self._dec = decrease  # 分解速度

        self._c = numpy.array([
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0],
        ]) * self._diffusion / (ds * ds)

        self._panels = []
        if model is not None:
            for y in range(0, self._ny):
                for x in range(0, self._nx):
                    px = ((x / (self._nx - 1.0)) - 0.5) * self._size * self._nx + self._px
                    py = ((y / (self._ny - 1.0)) - 0.5) * self._size * self._ny + self._py
                    deco_geom = model.add_deco_geom(mujoco.mjtGeom.mjGEOM_PLANE)
                    deco_geom.set_pos([px, py, 0.05])
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
        ix = ((x - self._px) / (self._size * self._nx) + 0.5) * (self._nx - 1.0)
        iy = ((y - self._py) / (self._size * self._ny) + 0.5) * (self._ny - 1.0)
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

    def add_liquid(self, x: float, y: float, value: float) -> None:
        """
        液体の量を表す指定した座標の付近の複数の要素に重みづけされた指定した量を加える．
        :param x: MuJoCo上のX座標．
        :param y: MuJoCo上のY座標．
        :param value: 加算する値．
        :return:
        """
        dico = self._calc_dico(x, y)
        for ix, iy, e in dico:
            if 0 <= ix < self._nx and 0 <= iy < self._ny:
                self._liquid[iy, ix] += value * e

    def set_liquid(self, x: float, y: float, value: float) -> None:
        dico = self._calc_dico(x, y)
        for ix, iy, e in dico:
            if 0 <= ix < self._nx and 0 <= iy < self._ny:
                self._liquid[iy, ix] = value * e

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

    def _euler_update(self, dt: float):
        """
        オイラー法による計算
        :param dt: 刻み幅
        :return: None
        """
        dif_liquid = numpy.minimum(self._liquid, (self._sv - self._gas) * self._eva * dt)
        dif_gas = dif_liquid + (scipy.signal.convolve2d(self._gas, self._c, "same") - self._gas * self._dec) * dt
        self._liquid -= dif_liquid
        self._gas += dif_gas

    def _rk_update(self, dt: float):
        """
        ルンゲ＝クッタ法による計算
        :param dt: 刻み幅
        :return: None
        """
        k1 = (self._sv - self._gas) * self._eva * dt
        k2 = 1.0 + 0.5 * self._eva * dt
        k3 = (1.0 + 0.5 * (1.0 + 0.5 * self._eva * dt)) * self._eva * dt
        k4 = (1.0 + (1.0 + 0.5 * (1.0 + 0.5 * self._eva * dt)) * self._eva * dt) * self._eva * dt
        dif_liquid = numpy.minimum(self._liquid * dt, k1 * (1.0 + 2.0 * k2 + 2.0 * k3 + k4) * 0.166666666666667)

        # 計算コスト削減のため，k2,k3,k4の拡散項の計算でk1の値を流用しています．
        # 検証していませんが，おそらく厳密には拡散項も計算しなおさないといけません．
        # しかし，k1の値を流用することでk2,k3,k4を行列の計算からスカラーの計算に変えられるため劇的に高速化できます．
        k1 = (scipy.signal.convolve2d(self._gas, self._c, "same") - self._gas * self._dec) * dt
        k2 = (1.0 - 0.5 * self._dec * dt)
        k3 = (1.0 - 0.5 * (1.0 - 0.5 * self._dec * dt) * self._dec * dt)
        k4 = (1.0 - (1.0 - 0.5 * (1.0 - 0.5 * self._dec * dt) * self._dec * dt) * self._dec * dt)
        dif_gas = dif_liquid + k1 * (1.0 + 2.0 * k2 + 2.0 * k3 + k4) * 0.166666666666667

        self._liquid -= dif_liquid
        self._gas += dif_gas

    def update_cells(self, dt: float = 0.03333):
        self._rk_update(dt)
        # self._euler_update(dt)

    def update_panels(self, func=lambda x: x):
        if len(self._panels) != 0:
            for p, g in zip(self._panels, numpy.clip(self._gas / 10.0, 0.0, 1.0).ravel()):
                v = func(g)
                color = colorsys.hsv_to_rgb(0.66 * (1.0 - v), 1.0, 1.0)
                p.set_rgba((color[0], color[1], color[2], 0.7))

    def get_gas_grad(self, pos: numpy.ndarray, direction: numpy.ndarray) -> numpy.ndarray:
        h_direction = numpy.array([
            numpy.cos(numpy.radians(-90)) * direction[0] - numpy.sin(numpy.radians(-90)) * direction[1],
            numpy.sin(numpy.radians(-90)) * direction[0] + numpy.cos(numpy.radians(-90)) * direction[1]
        ])
        v1_pos = pos[0:2] + self._size * direction
        v2_pos = pos[0:2] - self._size * direction
        h1_pos = pos[0:2] + self._size * h_direction
        h2_pos = pos[0:2] - self._size * h_direction
        v1 = self.get_gas(v1_pos[0], v1_pos[1])
        v2 = self.get_gas(v2_pos[0], v2_pos[1])
        h1 = self.get_gas(h1_pos[0], h1_pos[1])
        h2 = self.get_gas(h2_pos[0], h2_pos[1])
        return numpy.array([(v1 - v2) * 0.5 / self._ds, (h1 - h2) * 0.5 / self._ds])
