import numpy as np
import tkinter as tk

import mujoco

from torch import nn

import utils
from viewer import App, ViewerHandler
from environment import gen_xml




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
            for j, (id_, dist) in enumerate(zip(self.measure._intersected_id, self.measure._intersected_dist)):
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
