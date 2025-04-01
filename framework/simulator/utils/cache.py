import abc

from libs.mujoco_builder import WorldClock


class Cache(metaclass=abc.ABCMeta):
    def __init__(self, timer: WorldClock):
        self._update_time = None
        self._timer = timer

    @abc.abstractmethod
    def _update(self):
        raise NotImplemented

    def update(self):
        t = self._timer.get()
        if self._update_time == t:
            return
        self._update_time = t
        self._update()
