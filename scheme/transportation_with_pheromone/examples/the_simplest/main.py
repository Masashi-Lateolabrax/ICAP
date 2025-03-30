from prelude import framework

import numpy as np

from brain import Brain
from gui import App


class Loss(framework.interfaceis.Loss):
    def __init__(self, settings: framework.Settings):
        self.settings = settings

        self.sigma_nest_and_food = 1
        self.sigma_robot_and_food = 1

        self.gain_nest_and_food = 1
        self.gain_robot_and_food = 1

    def calc_loss(self, nest_pos: np.ndarray, robot_pos: np.ndarray, food_pos: np.ndarray) -> float:
        dist_r = np.linalg.norm(robot_pos[:, :2] - food_pos, axis=1)
        loss_r = np.average(np.exp(-dist_r / self.sigma_robot_and_food))

        dist_n = np.linalg.norm(nest_pos - food_pos, axis=1)
        loss_n = np.average(np.exp(-dist_n / self.sigma_nest_and_food))

        return self.gain_robot_and_food * loss_r + self.gain_nest_and_food * loss_n


def main():
    settings = framework.Settings()

    settings.CMAES.LOSS = Loss(settings)
    settings.Simulation.RENDER_HEIGHT = 800
    settings.Simulation.RENDER_WIDTH = 800

    brain = Brain()

    para = framework.Parameters(brain)
    task = framework.TaskGenerator(settings).generate(para, debug=False)

    app = App(settings, task, 3000)
    app.mainloop()


if __name__ == '__main__':
    main()
