import mujoco
import numpy as np

import utils
from viewer import App
from environment import gen_xml


class Main:
    def __init__(self):
        self.pos = np.zeros((3,))
        self.prev_pos = np.zeros((3,))
        self.time = 0

    def step(self, _m: mujoco.MjModel, d: mujoco.MjData):
        self.time += 1

        act_rot = d.actuator("bot1.act.rot")
        act_move = d.actuator("bot1.act.move")

        if self.time > 3000:
            act_move.ctrl[0] = -1.5
        elif self.time > 2500:
            act_rot.ctrl[0] = 0
        elif self.time > 2000:
            act_rot.ctrl[0] = 1.5
        elif self.time > 1500:
            act_move.ctrl[0] = 0
        elif self.time > 1000:
            act_move.ctrl[0] = 1.5
        elif self.time > 500:
            act_rot.ctrl[0] = 0
        elif self.time > 0:
            act_rot.ctrl[0] = 1.5


def entry_point():
    bot_pos = utils.generate_circles(3, 0.3, 5)
    goal_pos = utils.generate_circles(3, 0.4, 5)
    main = Main()
    app = App(gen_xml(bot_pos, goal_pos), 640, 480, main.step)
    app.mainloop()


if __name__ == '__main__':
    entry_point()
