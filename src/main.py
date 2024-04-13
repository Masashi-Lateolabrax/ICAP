import numpy as np
import tkinter as tk

import mujoco

from torch import nn

import utils
from viewer import App, ViewerHandler
from environment import gen_xml


class DistanceMeasure:
    def __init__(self, n):
        self.n = n
        self.intersected_id = np.zeros((n,), dtype=np.int32)
        self.intersected_dist = np.zeros((n,), dtype=np.float64)

        unit_quat = np.zeros((4,))
        mujoco.mju_axisAngle2Quat(unit_quat, [0, 0, 1], -mujoco.mjPI * 2 / n)

        buf1 = np.array([0, -1, 0], dtype=np.float64)
        buf2 = np.zeros((3,), dtype=np.float64)
        vecs = np.zeros((n, 3))
        for i in range(n):
            if i % 2 == 0:
                a = buf1
                b = buf2
            else:
                a = buf2
                b = buf1
            np.copyto(vecs[i], a)
            print(vecs[i])
            mujoco.mju_rotVecQuat(b, a, unit_quat)
        self.vecs = vecs.reshape((n * 3,)).copy()

    def measure(self, m: mujoco.MjModel, d: mujoco.MjData, botname, exclude_id: int):
        bot_body = d.body(botname)
        center_point = bot_body.xpos
        center_point[2] *= 0.5

        mujoco.mj_multiRay(
            m, d,
            pnt=center_point,
            vec=self.vecs,
            geomgroup=None,
            flg_static=1,
            bodyexclude=exclude_id,
            geomid=self.intersected_id,
            dist=self.intersected_dist,
            nray=self.n,
            cutoff=100,
        )


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
        self.renderer: mujoco.Renderer | None = None

        n = 64
        self.measure = DistanceMeasure(n)
        self.sight = np.zeros((1, n, 3), dtype=np.uint8)

    def customize_tk(self, tk_top: tk.Tk):
        pass

    def step(self, m: mujoco.MjModel, d: mujoco.MjData, gui: tk.Tk):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(m, 240, 240)

        cam_name = gui.children["!infoview"].camera_names.get()

        for i in range(self.nbot):
            id_ = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"bot{i}.body")
            act_rot = d.actuator(f"bot{i}.act.rot")
            act_move = d.actuator(f"bot{i}.act.move")

            self.measure.measure(m, d, f"bot{i}.body", id_)

            self.sight.fill(0)
            color = np.zeros((3,), dtype=np.float32)
            for j, (id_, dist) in enumerate(zip(self.measure.intersected_id, self.measure.intersected_dist)):
                if id_ < 0:
                    continue

                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, id_)
                if "bot" in name:
                    color[:] = [255, 255, 0]
                elif "goal" in name:
                    color[:] = [0, 255, 0]

                self.sight[0, j, :] = color / ((dist * 0.1 + 1) ** 2)

            if cam_name == f"bot{i}.camera":
                gui.children["!lidarview"].render(self.sight)


def entry_point():
    bot_pos = utils.generate_circles(5, 0.3, 5)
    goal_pos = utils.generate_circles(5, 0.4, 5)
    app = App(
        gen_xml(bot_pos, goal_pos),
        250, 250,
        Main(5)
    )
    app.mainloop()


if __name__ == '__main__':
    entry_point()
