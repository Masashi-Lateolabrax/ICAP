import tkinter as tk
import os

import numpy as np
from mujoco.renderer import Renderer

from mujoco_xml_generator.utils import MuJoCoView

import framework

from logger import Logger
from settings import Settings, set_positions
from brain import BrainBuilder


class Simulator:
    def __init__(self, settings: Settings, para):
        self.settings = settings

        brain_builder = BrainBuilder(settings)
        task_generator = framework.TaskGenerator(settings, brain_builder)
        self.task = task_generator.generate(para, debug=True)

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


class App(tk.Tk):
    WIDTH = 640
    HEIGHT = 480

    def __init__(self, settings: Settings, para):
        tk.Tk.__init__(self)

        self.simulator = Simulator(settings, para)

        self.renderer = Renderer(self.simulator.model, self.HEIGHT, self.WIDTH)

        self.m_view = MuJoCoView(self, self.WIDTH, self.HEIGHT)
        self.m_view.enable_input()
        self.m_view.pack()

        self.after(1, self.step)

    def step(self):
        self.simulator.step()
        self.m_view.render(self.simulator.data, self.renderer)


def main():
    save_dir = "20250428_022207-80edf54b"
    log_file_name = "result.pkl"

    logger = Logger.load(os.path.join(save_dir, log_file_name))

    settings = Settings()
    set_positions(settings)

    app = App(settings, logger[10].max_ind)
    app.mainloop()


if __name__ == '__main__':
    main()
