import tkinter as tk
import tkinter.ttk as ttk

import mujoco
import numpy as np
from PIL import Image as PILImage, ImageTk as PILImageTk

from mujoco_xml_generator.utils import FPSManager, MuJoCoView

from lib.utils import load_parameter

from experiment import Settings, TaskGenerator, Task, RobotDebugBuf


class InputView(tk.Frame):
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

    def render(self, buf: RobotDebugBuf):
        w = int(self.canvas_shape[1] / buf.input.shape[1] + 0.5)
        buf = np.repeat(buf.input, w, axis=1)

        h = int(self.canvas_shape[0] / buf.shape[0] + 0.5)
        buf = np.repeat(buf, h, axis=0)

        buf = PILImage.fromarray(buf)
        self.tkimg_buf.paste(buf)


class InfoView(tk.Frame):
    def __init__(self, master, cnf=None, **kw):
        if cnf is None:
            cnf = {}
        super().__init__(master, cnf, **kw)

        self.do_simulate = False

        def switch_move(x):
            self.do_simulate = x

        tk.Button(self, text="start", command=lambda: switch_move(True)).pack()
        tk.Button(self, text="stop", command=lambda: switch_move(False)).pack()

        self.interval_label = tk.Label(self)
        self.interval_label.pack()

        self.skip_rate_label = tk.Label(self)
        self.skip_rate_label.pack()

        self.evaluation_label = tk.Label(self)
        self.evaluation_label.pack()

        self.camera_lookat_label = tk.Label(self)
        self.camera_lookat_label.pack()

        self.camera_distance_label = tk.Label(self)
        self.camera_distance_label.pack()

        targets = [str(i) for i in range(Settings.Task.NUM_ROBOT)]
        self.target_ids = ttk.Combobox(self, values=targets, state="readonly")
        self.target_ids.pack()


class App(tk.Tk):
    def __init__(self, width, height, task: Task, manage_fps: bool = True):
        super().__init__()
        self.resizable(False, False)

        self._task = task

        self._renderer = mujoco.Renderer(
            self._task.get_model(), height, width, max_geom=Settings.Display.MAX_GEOM
        )

        self._manage_fps = manage_fps
        self._fps_manager = FPSManager(60)

        self.info_view = InfoView(self)
        self.info_view.grid(row=0, column=0, rowspan=2)

        self._mujoco_view = MuJoCoView(self, width, height)
        self._mujoco_view.enable_input()
        self._mujoco_view.grid(row=0, column=1)

        self._input_view = InputView(self, width, 50)
        self._input_view.grid(row=1, column=1, columnspan=3)

        self._input_buf = RobotDebugBuf()

        self.after(1, self.step)

    def step(self):
        interval = self._fps_manager.calc_interval()

        self._fps_manager.record_start()
        timestep = Settings.Simulation.TIMESTEP - 0.001
        do_rendering = self._fps_manager.render_or_not(timestep) or (not self._manage_fps)

        self._task.enable_updating_panels(do_rendering)

        target = self.info_view.target_ids.get()

        if do_rendering and target != "":
            target = int(target)
            self._task.set_input_buf(target, self._input_buf)

        evaluation = 0
        if self.info_view.do_simulate:
            evaluation = self._task.calc_step()

        if do_rendering:
            self._mujoco_view.render(self._task.get_data(), self._renderer, dummy_geoms=self._task.get_dummies())
            self._input_view.render(self._input_buf)

        self.info_view.interval_label.config(text=f"interval : {interval:.5f}")
        self.info_view.skip_rate_label.config(text=f"skip rate : {self._fps_manager.skip_rate:.5f}")
        self.info_view.evaluation_label.config(text=f"evaluation : {evaluation:10.5f}")
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


def viewer():
    working_directory = ""
    git_hash = None

    task_generator = TaskGenerator()
    dim = task_generator.get_dim()
    print(f"dim: {dim}")

    para = load_parameter(
        dim=task_generator.get_dim(),
        working_directory=working_directory,
        git_hash=git_hash,
        queue_index=-1,
    )

    task = task_generator.generate(para, True)
    app = App(500, 500, task, True)
    app.mainloop()


if __name__ == '__main__':
    viewer()
