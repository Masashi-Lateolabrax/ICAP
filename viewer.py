import cv2
import numpy
import glfw
import mujoco
import OpenGL.GL as gl

import studyLib.wrap_mjc as wm


class Recorder:
    def __init__(self, filepath: str, fps: int, video_size: (int, int)):
        codec = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        self.writer = cv2.VideoWriter(filepath, codec, fps, video_size)
        self.width = video_size[0]
        self.height = video_size[1]
        self.buffer = numpy.zeros(0)

    def __del__(self):
        self.writer.release()

    def record(self, viewer):
        w, h = viewer.get_size()

        if self.buffer.size == 0:
            self.buffer = numpy.zeros((h, w, 3), dtype="uint8")

        viewer.read_pixels((0, 0), (w, h), self.buffer)

        self.writer.write(
            cv2.resize(numpy.flipud(self.buffer), dsize=(self.width, self.height), interpolation=cv2.INTER_NEAREST)
        )

    def release(self):
        self.writer.release()


window_count = 0
glfw.terminate()


class Viewer:
    def __init__(self, width: int, height: int, title: str = "STUDY"):
        global window_count

        if window_count == 0:
            glfw.init()

        self.window = glfw.create_window(width, height, title, None, None)
        window_count += 1
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

    def __del__(self):
        global window_count
        window_count -= 1
        if window_count == 0:
            glfw.terminate()

    def get_size(self) -> (int, int):
        return glfw.get_framebuffer_size(self.window)

    def render(self, model: wm.WrappedModel, cam: wm.Camera = None, rect: (int, int, int, int) = None) -> bool:
        if glfw.window_should_close(self.window):
            return False

        ctx = model.get_ctx()
        scn = model.get_scene(cam)
        if rect is None:
            (width, height) = self.get_size()
            rect = mujoco.MjrRect(0, 0, width, height)
        else:
            rect = mujoco.MjrRect(rect[0], rect[1], rect[2], rect[3])

        glfw.make_context_current(self.window)
        mujoco.mjr_render(rect, scn, ctx)

        return True

    def flush(self) -> bool:
        if glfw.window_should_close(self.window):
            return False
        glfw.swap_buffers(self.window)
        glfw.poll_events()
        return True

    def read_pixels(self, upper_left: (int, int), lower_right: (int, int), buffer: numpy.ndarray):
        gl.glReadBuffer(gl.GL_BACK)
        gl.glReadPixels(
            upper_left[0], upper_left[1], lower_right[0], lower_right[1],
            gl.GL_BGR,
            gl.GL_UNSIGNED_BYTE,
            buffer
        )
