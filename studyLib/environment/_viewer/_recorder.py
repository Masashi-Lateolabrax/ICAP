import cv2
import numpy
import _viewer as vi
import _recorder_interface as rec_i


class Recorder(rec_i.RecorderInterface):
    def __init__(self, filepath: str, fps: int, video_size: (int, int)):
        codec = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        self.writer = cv2.VideoWriter(filepath, codec, fps, video_size)
        self.width = video_size[0]
        self.height = video_size[1]
        self.buffer = numpy.zeros(0)

    def __del__(self):
        self.writer.release()

    def release(self):
        self.writer.release()

    def record(self, viewer: vi.Viewer):
        w, h = viewer.get_size()

        if self.buffer.size == 0:
            self.buffer = numpy.zeros((h, w, 3), dtype="uint8")

        viewer.read_pixels((0, 0), (w, h), self.buffer)

        self.writer.write(
            cv2.resize(numpy.flipud(self.buffer), dsize=(self.width, self.height), interpolation=cv2.INTER_NEAREST)
        )
