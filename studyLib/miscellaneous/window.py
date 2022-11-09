import glfw
import cv2
from OpenGL import GL as gl
import numpy
import mujoco

from studyLib.wrap_mjc import WrappedModel
from studyLib.wrap_mjc import Camera

window_count = 0
glfw.terminate()


class Recorder:
    def __init__(self, filepath: str, fps: int, width: int, height: int):
        codec = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        self.writer = cv2.VideoWriter(filepath, codec, fps, (width, height))
        self.width = width
        self.height = height
        self.buffer = numpy.zeros(0)

    def __del__(self):
        self.writer.release()

    def record(self, window):
        (width, height) = glfw.get_framebuffer_size(window)
        if len(self.buffer) != width * height * 3:
            self.buffer = numpy.ndarray((height, width, 3), numpy.uint8)

        gl.glReadBuffer(gl.GL_BACK)
        gl.glReadPixels(0, 0, width, height, gl.GL_BGR, gl.GL_UNSIGNED_BYTE, self.buffer)

        self.writer.write(
            cv2.resize(numpy.flipud(self.buffer), dsize=(self.width, self.height), interpolation=cv2.INTER_NEAREST)
        )


class Window:
    def __init__(self, width: int, height: int, title: str = "STUDY"):
        global window_count

        if window_count == 0:
            glfw.init()

        self.window = glfw.create_window(width, height, title, None, None)
        window_count += 1
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.recorder: Recorder = None

    def __del__(self):
        global window_count
        window_count -= 1
        if window_count == 0:
            glfw.terminate()

    def get_size(self) -> (int, int):
        return glfw.get_framebuffer_size(self.window)

    def set_recorder(self, recorder: Recorder):
        self.recorder = recorder

    def render(self, model: WrappedModel, cam: Camera = None) -> bool:
        if glfw.window_should_close(self.window):
            return False

        ctx = model.get_ctx()
        scn = model.get_scene(cam)
        (width, height) = self.get_size()
        rect = mujoco.MjrRect(0, 0, width, height)

        glfw.make_context_current(self.window)
        mujoco.mjr_render(rect, scn, ctx)

        return True

    def flush(self) -> bool:
        if glfw.window_should_close(self.window):
            return False

        if not (self.recorder is None):
            self.recorder.record(self.window)

        glfw.swap_buffers(self.window)
        glfw.poll_events()

        return True
