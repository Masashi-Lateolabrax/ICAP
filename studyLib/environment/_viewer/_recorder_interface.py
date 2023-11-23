import abc
import _viewer as vi


class RecorderInterface(abc.ABCMeta):
    @abc.abstractmethod
    def record(self, viewer: vi.Viewer):
        raise NotImplementedError()

    @abc.abstractmethod
    def release(self):
        raise NotImplementedError()
