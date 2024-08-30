class PheromoneTank:
    def __init__(self, max_volume):
        self._max = max_volume
        self._tank = max_volume

    def secretion(self, volume) -> float:
        secretion = min(self._tank, volume)
        self._tank -= secretion
        return secretion

    def remain(self) -> float:
        return self._tank

    def charge(self, volume):
        self._tank = min(self._tank + volume, self._max)

    def fill(self):
        self._tank = self._max
