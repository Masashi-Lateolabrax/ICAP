import tkinter as tk

import numpy as np
import torch
from mujoco.renderer import Renderer

from mujoco_xml_generator.utils import MuJoCoView

import framework
from framework.simulator.objects.robot import RobotInput

from settings import Settings


class DummyBrain(framework.interfaces.BrainInterface):
    def __init__(self, right: np.ndarray, left: np.ndarray):
        super().__init__()
        self.right = right
        self.left = left

    def think(self, input_: RobotInput) -> torch.Tensor:
        return torch.tensor([self.right[0], self.left[0]])


class DummyBrainBuilder(framework.interfaces.BrainBuilder):
    def __init__(self, right: np.ndarray, left: np.ndarray):
        super().__init__()
        self.right = right
        self.left = left

    @staticmethod
    def get_dim() -> int:
        return 0

    def build(self, para):
        return DummyBrain(self.right, self.left)


class Simulator:
    def __init__(self, settings: Settings, right: np.ndarray, left: np.ndarray):
        self.settings = settings

        brain_builder = DummyBrainBuilder(right, left)
        task_generator = framework.TaskGenerator(settings, brain_builder)
        self.task = task_generator.generate(None, debug=True)

    @property
    def model(self):
        return self.task.get_model()

    @property
    def data(self):
        return self.task.get_data()

    def step(self):
        self.task.calc_step()
        dump: framework.Dump = self.task.get_dump_data()
        delta = dump.deltas[-1]

        r, n = self.settings.CMAES.LOSS.calc_elements(
            nest_pos=np.zeros(2),
            robot_pos=np.array(list(delta.robot_pos.values())),
            food_pos=delta.food_pos,
        )

        return r, n


class InfoFrame(tk.Frame):
    def __init__(self, master, cnf=None, **kw):
        if cnf is None:
            cnf = {}

        super().__init__(master, cnf, **kw)

        self.loss_r_label = tk.Label(self, text="Loss Robot")
        self.loss_r_label.pack()

        self.loss_n_label = tk.Label(self, text="Loss Nest")
        self.loss_n_label.pack()

    def update_info(self, loss_r: float, loss_n: float):
        self.loss_r_label.config(text=f"Loss Robot: {loss_r:.8f}")
        self.loss_n_label.config(text=f"Loss Nest: {loss_n:.8f}")


class App(tk.Tk):
    WIDTH = 640
    HEIGHT = 480

    def __init__(self, settings: Settings):
        tk.Tk.__init__(self)

        self.right = np.array([0.0])
        self.left = np.array([0.0])

        self.simulator = Simulator(settings, self.right, self.left)

        self.renderer = Renderer(self.simulator.model, self.HEIGHT, self.WIDTH)

        self.m_view = MuJoCoView(self, self.WIDTH, self.HEIGHT)
        self.m_view.enable_input()
        self.m_view.pack()

        self.info_frame = InfoFrame(self)
        self.info_frame.pack()

        self.bind("<KeyPress-w>", lambda e: self.on_key_press("forward"))
        self.bind("<KeyPress-s>", lambda e: self.on_key_press("stop"))
        self.bind("<KeyPress-d>", lambda e: self.on_key_press("left"))
        self.bind("<KeyPress-a>", lambda e: self.on_key_press("right"))

        self.after(1, self.step)

    def on_key_press(self, t):
        match t:
            case "forward":
                self.right[0] = 1
                self.left[0] = 1
            case "stop":
                self.right[0] = 0
                self.left[0] = 0
            case "left":
                self.right[0] = -1
                self.left[0] = 1
            case "right":
                self.right[0] = 1
                self.left[0] = -1

    def step(self):
        loss_r, loss_n = self.simulator.step()
        self.info_frame.update_info(loss_r, loss_n)
        self.m_view.render(self.simulator.data, self.renderer)
        self.after(1, self.step)


def main():
    settings = Settings()

    settings.Robot.POSITION = [(0, 0, 0)]
    settings.Food.POSITION = [
        (0, 6),
        (0, -6),
    ]
    settings.Robot.NUM = len(settings.Robot.POSITION)
    settings.Food.NUM = len(settings.Food.POSITION)

    settings.Food.REPLACEMENT = True

    settings.CMAES.LOSS.sigma_nest_and_food = 1
    settings.CMAES.LOSS.GAIN_NEST_AND_FOOD = 1
    settings.CMAES.LOSS.sigma_robot_and_food = 0.1
    settings.CMAES.LOSS.GAIN_ROBOT_AND_FOOD = 1

    app = App(settings)
    app.mainloop()


if __name__ == '__main__':
    main()
