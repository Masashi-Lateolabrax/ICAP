import time
import tkinter as tk
import numpy as np

from PIL import Image as PILImage, ImageTk as PILImageTk
import mujoco

import mujoco_xml_generator as mjc_gen
import mujoco_xml_generator.common as mjc_cmn
from mujoco_xml_generator import WorldBody, Body, body, Option


def gen_xml() -> str:
    generator = mjc_gen.Generator().add_children([
        Option(timestep=0.007),

        WorldBody().add_children([
            body.Geom(
                type_=mjc_cmn.GeomType.PLANE, pos=(0, 0, 0), size=(100, 100, 1), rgba=(1, 1, 1, 1)
            ),

            Body(
                pos=(0, 0, 10)
            ).add_children([
                body.Joint(type_=body.Joint.JointType.FREE),
                body.Geom(type_=mjc_cmn.GeomType.SPHERE, size=(1,))
            ]),

            Body(
                pos=(1.5, 1.5, 0.6)
            ).add_children([
                body.Joint(type_=body.Joint.JointType.FREE),
                body.Geom(type_=mjc_cmn.GeomType.BOX, size=(0.5, 0.5, 0.5))
            ])
        ])

    ])
    xml = generator.build()
    print(xml)
    return xml


class _MuJoCoProcess:
    def __init__(self, xml, width, height):
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height, width)

    def get_timestep(self) -> float:
        return self.model.opt.timestep

    def step(self, camera: mujoco.MjvCamera | None = None, out_img: np.ndarray | None = None):
        mujoco.mj_step(self.model, self.data)
        if camera is not None and out_img is not None:
            self.renderer.update_scene(self.data, camera)
            self.renderer.render(out=out_img)


class _InfoView(tk.Frame):
    def __init__(self, master=None, cnf=None, **kw):
        if cnf is None:
            cnf = {}
        super().__init__(master, cnf, **kw)

        self.interval_label = tk.Label(self)
        self.interval_label.pack()

        self.camera_lookat_label = tk.Label(self)
        self.camera_lookat_label.pack()

        self.camera_distance_label = tk.Label(self)
        self.camera_distance_label.pack()


class _MuJoCoView(tk.Frame):
    def __init__(self, master, width, height, cnf=None, **kw):
        if cnf is None:
            cnf = {}

        super().__init__(master, cnf, **kw)

        self.canvas = tk.Canvas(master=self, width=width, height=height)
        self.canvas.pack(expand=True)

        plk_img_buf = PILImage.fromarray(
            np.zeros((height, width, 3), dtype=np.uint8)
        )
        self.tkimg_buf = PILImageTk.PhotoImage(image=plk_img_buf)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tkimg_buf)

    def render(self, img_buf: np.ndarray):
        self.tkimg_buf.paste(
            PILImage.fromarray(img_buf)
        )


class App(tk.Tk):
    def __init__(self, xml, width, height):
        super().__init__()

        self._mouse_pos = [0, 0]
        self._mouse_left = False
        self._shift_key = False
        self._prev_step_start = 0.0

        self._mujoco = _MuJoCoProcess(xml, width, height)

        self.camera = mujoco.MjvCamera()
        self.img_buf = np.zeros((height, width, 3), dtype=np.uint8)

        self.resizable(False, False)

        self._info_frame = _InfoView(self)
        self._info_frame.grid(row=0, column=0)

        self._mujoco_view = _MuJoCoView(self, width, height)
        self._mujoco_view.grid(row=0, column=1)

        self.bind("<Motion>", self._mouse_move_handler)
        self.bind("<Shift-Motion>", self._mouse_shift_move_handler)
        self.bind("<MouseWheel>", lambda e: self._mouse_wheel_handler(e))
        self.bind("<ButtonPress-4>", lambda e: self._mouse_wheel_handler(e, "up"))
        self.bind("<ButtonPress-5>", lambda e: self._mouse_wheel_handler(e, "down"))
        self.bind("<Button-1>", lambda e: self._mouse_left_handler(e, "down"))
        self.bind("<ButtonRelease-1>", lambda e: self._mouse_left_handler(e, "up"))

        self.after(1, self.step)

    def _calc_camera_direction(self):
        quat = np.zeros((4,))
        v = np.zeros((3,))
        mujoco.mju_axisAngle2Quat(quat, [0, 1, 0], -mujoco.mjPI * self.camera.elevation / 180)
        mujoco.mju_rotVecQuat(v, [1, 0, 0], quat)
        mujoco.mju_axisAngle2Quat(quat, [0, 0, 1], mujoco.mjPI * self.camera.azimuth / 180)
        mujoco.mju_rotVecQuat(v, v, quat)
        return v

    def _calc_camera_up(self):
        quat = np.zeros((4,))
        v = np.zeros((3,))
        mujoco.mju_axisAngle2Quat(quat, [0, 1, 0], -mujoco.mjPI * self.camera.elevation / 180)
        mujoco.mju_rotVecQuat(v, [0, 0, 1], quat)
        mujoco.mju_axisAngle2Quat(quat, [0, 0, 1], mujoco.mjPI * self.camera.azimuth / 180)
        mujoco.mju_rotVecQuat(v, v, quat)
        return v

    def _mouse_wheel_handler(self, event, mode=None):
        sensitivity = 0.3

        v = self._calc_camera_direction()
        if mode == "up":
            self.camera.lookat += v * sensitivity
            return
        elif mode == "down":
            self.camera.lookat -= v * sensitivity
            return

    def _mouse_move_handler(self, event):
        dx = event.x - self._mouse_pos[0]
        dy = event.y - self._mouse_pos[1]
        self._mouse_pos[0] = event.x
        self._mouse_pos[1] = event.y

        if self._mouse_left:
            sensitivity = 0.1
            self.camera.elevation += dy * sensitivity
            self.camera.azimuth += dx * sensitivity

    def _mouse_shift_move_handler(self, event):
        dx = event.x - self._mouse_pos[0]
        dy = event.y - self._mouse_pos[1]
        self._mouse_pos[0] = event.x
        self._mouse_pos[1] = event.y

        if self._mouse_left:
            sensitivity = 0.01
            v = self._calc_camera_direction()
            up = self._calc_camera_up()
            left = np.cross(v, up)
            self.camera.lookat += up * sensitivity * dy
            self.camera.lookat -= left * sensitivity * dx

    def _mouse_left_handler(self, event, mode):
        self._mouse_left = mode == "down"

    def step(self, handler=lambda m, d: ()):
        step_start = time.time()
        interval = step_start - self._prev_step_start
        self._prev_step_start = step_start
        timestep = self._mujoco.get_timestep()

        self._mujoco.step(self.camera, self.img_buf)
        self._mujoco_view.render(self.img_buf)

        handler(self._mujoco.model, self._mujoco.data)

        self._info_frame.interval_label.config(text=f"interval : {interval:.5f}")
        self._info_frame.camera_lookat_label.config(
            text=f"camera lookat :\n {self.camera.lookat[0]:.3f}\n {self.camera.lookat[1]:.3f}\n {self.camera.lookat[2]:.3f}"
        )
        self._info_frame.camera_distance_label.config(text=f"camera distance : {self.camera.distance:.5f}")

        time_until_next_step = (timestep - (time.time() - step_start)) * 1000
        if time_until_next_step > 1.0:
            self.after(int(time_until_next_step + 0.5), self.step)
        else:
            self.after(1, self.step)


if __name__ == '__main__':
    def main():
        app = App(gen_xml(), 640, 480)
        app.mainloop()


    main()
