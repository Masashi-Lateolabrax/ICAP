class WorldClock:
    """
    MuJoCoのシミュレーション時間を共有するために使われるクラス
    """

    def __init__(self):
        self._time = 0

    def get(self):
        """
        MuJoCoが何ステップ計算されたかを返す
        """
        return self._time
