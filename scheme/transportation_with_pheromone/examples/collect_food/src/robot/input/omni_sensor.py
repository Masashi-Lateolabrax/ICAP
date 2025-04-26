import unittest
from unittest.mock import Mock

import numpy as np
from mujoco._structs import _MjDataSiteViews


class OmniSensor:
    """
    三角関数を用いた全方位センサです．

    登録された知覚対象のMjDataSiteViewsから対象の位置を取得し，センサーの出力を計算します．
    """

    def __init__(self, gain: float, offset: float, target_sites: list[_MjDataSiteViews]):
        """
        OmniSensorのコンストラクタ。

        Args:
            gain (float): センサーのゲイン値。
            offset (float): センサーのオフセット値。
            target_sites (list[_MjDataSiteViews]): ターゲットサイトのリスト。
        """
        self.gain = gain
        self.offset = offset
        self.targets: list[_MjDataSiteViews] = target_sites
        self.target_positions = np.zeros((len(self.targets), 2))

    def _update_positions(self):
        """
        ターゲットサイトの位置を更新します。
        """
        for i, p in enumerate(self.targets):
            self.target_positions[i] = p.xpos[0:2]

    def get(self, bot_direction, bot_pos) -> np.ndarray:
        """
        センサーの出力値を計算します。

        Args:
            bot_direction (array-like): ロボットの方向ベクトル。
            bot_pos (array-like): ロボットの位置ベクトル。

        Returns:
            array: センサーの出力値。2つの要素からなり，それぞれロボットから見て横方向と前方向の出力値を表します。
        """
        if len(self.targets) == 0:
            return np.zeros(2)

        direction = np.array([
            [bot_direction[1], -bot_direction[0]],
            [bot_direction[0], bot_direction[1]],
        ])
        self._update_positions()

        sub = self.target_positions - bot_pos[0:2]
        distance = np.linalg.norm(sub, axis=1)
        if np.any(distance == 0):
            # print("distance array contains 0")
            return np.zeros(2)

        sub = sub.T / distance
        trigono_components = np.dot(direction, sub)
        scaled_distance = self.gain * np.maximum(distance - self.offset, 0)
        res = np.dot(
            trigono_components,
            np.reciprocal(scaled_distance + 1),
        )
        res /= self.target_positions.shape[0]

        return res


class _TestOmniSensor(unittest.TestCase):
    @staticmethod
    def _create_target_sites(pos: list[tuple[float, float, float]]):
        target_sites: list[_MjDataSiteViews] = []
        for p in pos:
            site = Mock()
            site.xpos = np.array(p)
            target_sites.append(site)
        return target_sites

    def test_x0y10(self):
        sensor = OmniSensor(
            1, 0,
            _TestOmniSensor._create_target_sites([
                (0, 10, 0)
            ])
        )
        res = sensor.get(np.array([0, 1]), np.array([0, 0, 0]))

        assert res[0] == 0
        assert abs(res[1] - 0.09090) < 1e-5

    def test_x10y0(self):
        sensor = OmniSensor(
            1, 0,
            _TestOmniSensor._create_target_sites([
                (10, 0, 0)
            ])
        )
        res = sensor.get(np.array([0, 1]), np.array([0, 0, 0]))

        assert abs(res[0] - 0.09090) < 1e-5
        assert res[1] == 0

    def test_xN5yN5(self):
        sensor = OmniSensor(
            1, 0,
            _TestOmniSensor._create_target_sites([
                (-5, -5, 0)
            ])
        )
        res = sensor.get(np.array([0, 1]), np.array([0, 0, 0]))

        assert abs(res[0] - -0.087610066) < 1e-5
        assert abs(res[1] - -0.087610066) < 1e-5

    def test_x0y5_x5y0(self):
        sensor = OmniSensor(
            1, 0,
            _TestOmniSensor._create_target_sites([
                (0, 5, 0), (5, 0, 0)
            ])
        )
        res = sensor.get(np.array([0, 1]), np.array([0, 0, 0]))

        # (0,5) -> [cos 90  sin 90] -> [0  1]
        # (5,0) -> [cos 0  sin 0] -> [1  0]
        # ( [0  1]/(5+1) + [1  0]/(5+1) )/2 = [0.083333333  0.083333333]

        assert abs(res[0] - 0.083333333) < 1e-5
        assert abs(res[1] - 0.083333333) < 1e-5

    def test_x0y10_x5y0(self):
        sensor = OmniSensor(
            1, 0,
            _TestOmniSensor._create_target_sites([
                (0, 10, 0), (5, 0, 0)
            ])
        )
        res = sensor.get(np.array([0, 1]), np.array([0, 0, 0]))

        # (0,10) -> [cos 90  sin 90]/(10+1) -> [0  1]/11 -> [0  0.090909091]
        # (5,0) -> [cos 0  sin 0]/(5+1) -> [1  0]/6 -> [0.166666667  0]
        # ( [0  0.090909091] + [0.166666667  0] )/2 = [0.083333334  0.045454546]

        assert abs(res[0] - 0.083333334) < 1e-5
        assert abs(res[1] - 0.045454546) < 1e-5

    def test_r150d1_r300d1(self):
        sensor = OmniSensor(
            1, 0,
            _TestOmniSensor._create_target_sites([
                (-0.866025404, 0.5, 0), (0.5, -0.866025404, 0)
            ])
        )
        res = sensor.get(np.array([0, 1]), np.array([0, 0, 0]))

        # (-0.86602, 0.5) -> [cos 150  sin 150]/(1+1) -> [−0.86602  0.5]/2 -> [−0.433012702  0.25]
        # (0.5, -0.86602) -> [cos 300  sin 300]/(1+1) -> [0.5  −0.86602]/2 -> [0.25  −0.433012702]
        # ( [−0.433012702  0.25] + [0.25  −0.433012702] )/2 = [−0.091506351  −0.091506351]

        assert abs(res[0] - -0.091506351) < 1e-5
        assert abs(res[1] - -0.091506351) < 1e-5

    def test_r150d3_r300d5(self):
        sensor = OmniSensor(
            1, 0,
            _TestOmniSensor._create_target_sites([
                (-2.598076212, 1.5, 0), (2.5, -4.33012702, 0)
            ])
        )
        res = sensor.get(np.array([0, 1]), np.array([0, 0, 0]))

        # (-2.598076212, 1.5) -> [cos 150  sin 150]/(3+1) -> [−0.86602  0.5]/4 -> [−0.21650  0.125]
        # (2.5, -4.33012702) -> [cos 300  sin 300]/(5+1) -> [0.5  −0.86602]/6 -> [0.08333  −0.14433]
        # ( [−0.21650  0.125] + [0.08333  −0.14433] )/2 = [−0.066585  −0.009665]

        assert abs(res[0] - -0.066585) < 1e-5
        assert abs(res[1] - -0.009665) < 1e-5

    def test_rnh_x0y10(self):
        sensor = OmniSensor(
            1, 0,
            _TestOmniSensor._create_target_sites([
                (0, 10, 0)
            ])
        )
        res = sensor.get(np.array([-1, 0]), np.array([0, 0, 0]))

        assert abs(res[0] - 0.09090) < 1e-5
        assert abs(res[1] - 0) < 1e-5

    def test_rnh_x10y0(self):
        sensor = OmniSensor(
            1, 0,
            _TestOmniSensor._create_target_sites([
                (10, 0, 0)
            ])
        )
        res = sensor.get(np.array([-1, 0]), np.array([0, 0, 0]))

        assert abs(res[0] - 0) < 1e-5
        assert abs(res[1] - -0.09090) < 1e-5
