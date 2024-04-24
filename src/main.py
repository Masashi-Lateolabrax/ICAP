import tkinter as tk

import mujoco

from viewer import App, ViewerHandler
from distance_measure import DistanceMeasure
from brain import NeuralNetwork
from optimizer import CMAES
from environment import ECreator
from robot import Robot


class Main(ViewerHandler):
    def __init__(self, nbot: int, para):
        self.nbot = nbot
        self.bots = []

        self._renderer: mujoco.Renderer | None = None

        self._measure = DistanceMeasure(64)

        self._brain = NeuralNetwork()
        self._brain.load_para(para)

        self._time = 0

    def customize_tk(self, tk_top: tk.Tk):
        pass

    def step(self, m: mujoco.MjModel, d: mujoco.MjData, gui: tk.Tk):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(m, 240, 240)
        if len(self.bots) == 0:
            self.bots = [Robot(m, d, i, self._brain) for i in range(self.nbot)]

        cam_name = gui.children["!infoview"].camera_names.get()

        for bot in self.bots:
            direction = bot.calc_direction()
            sight = self._measure.measure_with_img(
                m, d,
                bot.bot_body_id, bot.bot_body,
                direction
            )
            bot.exec(sight)

            if cam_name == bot.cam_name:
                gui.children["!lidarview"].render(sight)


def entry_point():
    n = 1

    dim = sum([p.numel() for p in NeuralNetwork().parameters() if p.requires_grad])
    cmaes = CMAES(dim, 3, 3, max_thread=1)
    env_creator = ECreator(n, n)
    cmaes.optimize(env_creator)
    cmaes.get_history().save("history.npz")

    import utils
    from environment import gen_xml
    bot_pos = utils.generate_circles(n, 0.3 * 1.01, 5)
    goal_pos = utils.generate_circles(n, 0.4 * 1.01, 5)
    app = App(
        gen_xml(bot_pos, goal_pos),
        500, 500,
        Main(1, cmaes.get_best_para())
    )
    app.mainloop()


if __name__ == '__main__':
    entry_point()
