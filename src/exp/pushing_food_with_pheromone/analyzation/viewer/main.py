import tkinter as tk
import tkinter.ttk as ttk

import mujoco
import numpy as np
from PIL import Image as PILImage, ImageTk as PILImageTk

from mujoco_xml_generator.utils import FPSManager, MuJoCoView

from src.exp.pushing_food_with_pheromone.settings import HyperParameters, Task, TaskGenerator
from lib.utils import load_parameter, get_head_hash


class InputView(tk.Frame):
    RED = np.array([255, 0, 0], dtype=np.float32)
    BLUE = np.array([0, 0, 255], dtype=np.float32)
    TRANSITION = RED - BLUE

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
        colored_img = (InputView.BLUE + InputView.TRANSITION * (img + 1) * 0.5).astype(np.uint8)
        colored_img = np.expand_dims(colored_img, axis=0)
        x = int(self.canvas_shape[1] / colored_img.shape[1] + 0.5)
        buf = np.repeat(colored_img, x, axis=1)
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

        self.evaluation_label = tk.Label(self)
        self.evaluation_label.pack()

        self.camera_lookat_label = tk.Label(self)
        self.camera_lookat_label.pack()

        self.camera_distance_label = tk.Label(self)
        self.camera_distance_label.pack()

        self.camera_names = ttk.Combobox(self, values=cameras, state="readonly")
        self.camera_names.pack()


class App(tk.Tk):
    def __init__(self, width, height, task: Task, manage_fps: bool = True):
        super().__init__()

        self._task = task

        self._depth_img_buf = np.zeros((height, width), dtype=np.float32)

        self._manage_fps = manage_fps
        self._fps_manager = FPSManager(60)

        self._renderer = mujoco.Renderer(
            self._task.get_model(), height, width, max_geom=HyperParameters.Simulator.MAX_GEOM
        )
        self._renderer_for_depth = mujoco.Renderer(self._task.get_model(), height, width)
        self._renderer_for_depth.enable_depth_rendering()

        self.resizable(False, False)

        camera_names = [
            mujoco.mj_id2name(self._task.get_model(), mujoco.mjtObj.mjOBJ_CAMERA, i) for i in
            range(self._task.get_model().ncam)
        ]
        self.info_view = InfoView(self, camera_names)
        self.info_view.grid(row=0, column=0, rowspan=2)

        self._mujoco_view = MuJoCoView(self, width, height)
        self._mujoco_view.enable_input()
        self._mujoco_view.grid(row=0, column=1)

        self._depth_view = MuJoCoView(self, width, height)
        self._depth_view.grid(row=0, column=2)

        self._camera_view = MuJoCoView(self, width, height)
        self._camera_view.grid(row=0, column=3)

        self._input_view = InputView(self, width * 3, 50)
        self._input_view.grid(row=1, column=1, columnspan=3)

        self.after(1, self.step)

    def step(self):
        interval = self._fps_manager.calc_interval()

        self._fps_manager.record_start()
        timestep = HyperParameters.Simulator.TIMESTEP - 0.001
        do_rendering = self._fps_manager.render_or_not(timestep) or (not self._manage_fps)

        evaluation = 0
        if self.info_view.do_simulate:
            evaluation = self._task.calc_step()

        if do_rendering:
            cam_name = self.info_view.camera_names.get()
            if cam_name != "":
                self._camera_view.camera = cam_name
                self._depth_view.camera = cam_name

            self._mujoco_view.render(self._task.get_data(), self._renderer, dummy_geoms=self._task.get_dummies())
            self._camera_view.render(self._task.get_data(), self._renderer)
            self._depth_view.render(self._task.get_data(), self._renderer_for_depth, self._depth_img_buf)

            self._task.mujoco.last_exec_robot_index = -1
            for i, bot in enumerate(self._task.get_bots()):
                if cam_name != bot.cam_name:
                    continue
                self._task.mujoco.last_exec_robot_index = i
                input_ = self._task.mujoco.input_sight_buf.detach().numpy()
                self._input_view.render(input_)

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


def main():
    working_directory = "../../../../"

    task_generator = TaskGenerator()
    dim = task_generator.get_dim()
    print(f"dim: {dim}")

    para = load_parameter(
        dim=task_generator.get_dim(),
        working_directory=working_directory,
        git_hash=get_head_hash(),
        queue_index=-1,
    )

    task = task_generator.generate(para)
    app = App(500, 500, task, False)
    app.mainloop()


if __name__ == '__main__':
    main()
