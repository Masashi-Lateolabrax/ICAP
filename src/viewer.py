import abc
import tkinter as tk
import tkinter.ttk as ttk

import mujoco
import numpy as np
from PIL import Image as PILImage, ImageTk as PILImageTk

from mujoco_xml_generator.utils import FPSManager, MuJoCoView

from environment import gen_xml


def main():
    app = App(gen_xml([(0, 0, 0)], [(0, 0, 0)]), 640, 480)
    app.mainloop()


class ViewerHandler(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def customize_tk(self, tk_top: tk.Tk):
        pass

    def renderer(self):
        pass

    @abc.abstractmethod
    def step(self, model: mujoco.MjModel, data: mujoco.MjData, gui: tk.Tk):
        raise NotImplementedError


class DefaultHandler(ViewerHandler):
    def customize_tk(self, tk_top: tk.Tk):
        pass

    def step(self, model: mujoco.MjModel, data: mujoco.MjData, gui: tk.Tk):
        pass


class _MuJoCoProcess:
    def __init__(self, xml):
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

    def get_timestep(self) -> float:
        return self.model.opt.timestep

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def camera_names(self):
        return [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(self.model.ncam)]


class LidarView(tk.Frame):
    def __init__(self, master, width: int, height: int, cnf=None, **kw):
        if cnf is None:
            cnf = {}
        super().__init__(master, cnf, **kw)
        self.canvas_shape: tuple[int, int] = (height, width)
        self.canvas = tk.Canvas(self, width=width, height=height)
        self.canvas.pack()

        plk_img_buf = PILImage.fromarray(
            np.zeros((height, width, 3), dtype=np.uint8)
        )
        self.tkimg_buf = PILImageTk.PhotoImage(image=plk_img_buf)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tkimg_buf)

    def render(self, img: np.ndarray):
        x = int(self.canvas_shape[1] / img.shape[1] + 0.5)
        buf = np.repeat(img, x, axis=1)
        buf = np.repeat(buf, self.canvas_shape[0], axis=0)
        self.tkimg_buf.paste(
            PILImage.fromarray(buf)
        )


class InfoView(tk.Frame):
    def __init__(self, master, cameras: list[str], cnf=None, **kw):
        if cnf is None:
            cnf = {}
        super().__init__(master, cnf, **kw)

        def switch_move(x):
            self.do_simulate = x

        self.do_simulate = False
        tk.Button(self, text="start", command=lambda: switch_move(True)).pack()
        tk.Button(self, text="stop", command=lambda: switch_move(False)).pack()

        self.interval_label = tk.Label(self)
        self.interval_label.pack()

        self.skip_rate_label = tk.Label(self)
        self.skip_rate_label.pack()

        self.camera_lookat_label = tk.Label(self)
        self.camera_lookat_label.pack()

        self.camera_distance_label = tk.Label(self)
        self.camera_distance_label.pack()

        self.camera_names = ttk.Combobox(self, values=cameras, state="readonly")
        self.camera_names.pack()


class App(tk.Tk):
    def __init__(self, xml, width, height, handler: ViewerHandler = DefaultHandler()):
        super().__init__()

        if not isinstance(handler, ViewerHandler):
            raise "Please give an instance of ViewerHandler to the 'handler' argument."

        self._depth_img_buf = np.zeros((height, width), dtype=np.float32)

        self._mujoco = _MuJoCoProcess(xml)
        self._fps_manager = FPSManager(self._mujoco.get_timestep(), 60)

        self._renderer = mujoco.Renderer(self._mujoco.model, height, width)
        self._renderer_for_depth = mujoco.Renderer(self._mujoco.model, height, width)
        self._renderer_for_depth.enable_depth_rendering()

        self.resizable(False, False)

        self.info_view = InfoView(self, self._mujoco.camera_names())
        self.info_view.grid(row=0, column=0, rowspan=2)

        self._mujoco_view = MuJoCoView(self, width, height)
        self._mujoco_view.enable_input()
        self._mujoco_view.grid(row=0, column=1)

        self._depth_view = MuJoCoView(self, width, height)
        self._depth_view.grid(row=0, column=2)

        self._camera_view = MuJoCoView(self, width, height)
        self._camera_view.grid(row=0, column=3)

        self.lidar_view = LidarView(self, width * 3, 50)
        self.lidar_view.grid(row=1, column=1, columnspan=3)

        self.handler = handler
        self.handler.customize_tk(self)

        self.after(1, self.step)

    def step(self):
        interval = self._fps_manager.calc_interval()

        self._fps_manager.record_start()
        timestep = self._mujoco.get_timestep() - 0.001
        do_rendering = self._fps_manager.render_or_not(timestep)

        if self.info_view.do_simulate:
            self._mujoco.step()
            self.handler.step(self._mujoco.model, self._mujoco.data, self)

        if do_rendering:
            cam_name = self.info_view.camera_names.get()
            if cam_name != "":
                self._camera_view.camera = cam_name
                self._depth_view.camera = cam_name

            self._mujoco_view.render(self._mujoco.data, self._renderer)
            self._camera_view.render(self._mujoco.data, self._renderer)
            self._depth_view.render(self._mujoco.data, self._renderer_for_depth, self._depth_img_buf)

        self.info_view.interval_label.config(text=f"interval : {interval:.5f}")
        self.info_view.skip_rate_label.config(text=f"skip rate : {self._fps_manager.skip_rate:.5f}")
        view_camera = self._mujoco_view.camera
        self.info_view.camera_lookat_label.config(
            text=f"camera lookat :\n {view_camera.lookat[0]:.3f}\n {view_camera.lookat[1]:.3f}\n {view_camera.lookat[2]:.3f}"
        )
        self.info_view.camera_distance_label.config(text=f"camera distance : {view_camera.distance:.5f}")

        self._fps_manager.record_stop(do_rendering)

        time_until_next_step = (timestep - self._fps_manager.ave_interval) * 1000
        if time_until_next_step > 1.0:
            self.after(int(time_until_next_step), self.step)
        else:
            self.after(1, self.step)


if __name__ == '__main__':
    main()
