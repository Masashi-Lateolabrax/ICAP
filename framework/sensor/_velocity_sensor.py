import numpy as np

from ..prelude import *


class VelocitySensor(SensorInterface):
    """
    ロボットの速度を取得するセンサー。

    MuJoCo の velocimeter と gyro センサーから速度情報を取得し、
    ロボットのローカル座標系における線速度（XY平面）と角速度（Z軸周り）を返します。
    """

    def __init__(self, robot: RobotValues):
        """
        VelocitySensorのコンストラクタ。

        Args:
            robot (RobotValues): ロボットの状態を保持するオブジェクト。
        """
        self.robot = robot

    def get(self) -> np.ndarray:
        """
        ロボットの速度情報を取得します。

        MuJoCo の velocimeter と gyro センサーから直接データを取得します。
        velocimeter は 3D linear velocity (local frame) を返し、
        gyro は 3D angular velocity (local frame) を返します。

        Returns:
            np.ndarray: [v_x, v_y, ω_z] の3次元配列
                - v_x: ロボットのローカルX軸方向の速度
                - v_y: ロボットのローカルY軸方向の速度
                - ω_z: ロボットの角速度（Z軸周り）
        """

        velocimeter_data = self.robot.velocity
        gyro_data = self.robot.angular_velocity

        return np.array([
            velocimeter_data[0],
            velocimeter_data[1],
            gyro_data[2]
        ], dtype=np.float32)
