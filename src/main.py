import numpy as np
import tkinter as tk

import mujoco

from torch import nn

import utils
from viewer import App, ViewerHandler
from environment import gen_xml
from distance_measure import DistanceMeasure


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


class Main(ViewerHandler):
    def __init__(self, nbot: int):
        self.nbot = nbot
        self._renderer: mujoco.Renderer | None = None

        n = 64
        self._measure = DistanceMeasure(n)
        self._calc_sight_buf = np.zeros((1, n, 3), dtype=np.uint8)

        self._rot_buf = np.zeros((4, 1), dtype=np.float64)
        self._move_buf = np.zeros((3, 1), dtype=np.float64)
        self._time = 0

    def customize_tk(self, tk_top: tk.Tk):
        pass

    def calc_sight(self, m: mujoco.MjModel, d: mujoco.MjData, bot_body_id: int):
        ids, dists = self._measure.measure(m, d, bot_body_id)

        self._calc_sight_buf.fill(0)
        color = np.zeros((3,), dtype=np.float32)
        for j, (id_, dist) in enumerate(zip(ids, dists)):
            if id_ < 0:
                continue

            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, id_)
            if "bot" in name:
                color[:] = [255, 255, 0]
            elif "goal" in name:
                color[:] = [0, 255, 0]

            self._calc_sight_buf[0, j, :] = color / ((dist * 0.1 + 1) ** 2)

        return self._calc_sight_buf

    def step(self, m: mujoco.MjModel, d: mujoco.MjData, gui: tk.Tk):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(m, 240, 240)

        self._time += 1
        cam_name = gui.children["!infoview"].camera_names.get()

        for i in range(self.nbot):
            bot_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"bot{i}.body")
            act_rot = d.actuator(f"bot{i}.act.rot")
            act_move_x = d.actuator(f"bot{i}.act.pos_x")
            act_move_y = d.actuator(f"bot{i}.act.pos_y")
            sight = self.calc_sight(m, d, bot_body_id)

            # act_rot.ctrl[0] += 0.5 * m.opt.timestep
            # movement = 1.2 * m.opt.timestep

            if self._time > 1200:
                movement = 0.0
                rotation = 0.0
            elif self._time > 900:
                movement = 0.3
                rotation = 0.0
            elif self._time > 600:
                movement = 0.0
                rotation = -1.0
            elif self._time > 300:
                movement = 0.5
                rotation = 0.0
            else:
                movement = 0.0
                rotation = 1.0

            mujoco.mju_axisAngle2Quat(self._rot_buf, [0, 0, 1], act_rot.length)
            mujoco.mju_rotVecQuat(self._move_buf, [0, 1, 0], self._rot_buf)

            act_rot.ctrl[0] += 0.5 * rotation * m.opt.timestep
            act_move_x.ctrl[0] += 1.2 * self._move_buf[0] * movement * m.opt.timestep
            act_move_y.ctrl[0] += 1.2 * self._move_buf[1] * movement * m.opt.timestep

            if cam_name == f"bot{i}.camera":
                gui.children["!lidarview"].render(sight)


def entry_point():
    n = 1
    bot_pos = utils.generate_circles(n, 0.3 * 1.01, 5)
    goal_pos = utils.generate_circles(n, 0.4 * 1.01, 5)
    app = App(
        gen_xml([(0, 0, 0)], goal_pos),
        500, 500,
        Main(n)
    )
    app.mainloop()


if __name__ == '__main__':
    entry_point()
