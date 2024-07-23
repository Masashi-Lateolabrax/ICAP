import lib.optimizer as opt

from .settings import Settings
from .core import World, lidar_c, trigonometric
from ._task import Task


class TaskGenerator(opt.TaskGenerator):
    def __init__(self):
        self.safezone_pos = Settings.Task.SafeZone.POSITION()
        self.bot_pos = Settings.Task.Robot.POSITIONS()

    @staticmethod
    def get_dim():
        dim = 0
        if Settings.Task.NUM_ROBOT_LI > 0:
            dim += lidar_c.RobotC.get_dim()
        if Settings.Task.NUM_ROBOT_TR > 0:
            dim += trigonometric.RobotTR.get_dim()
        return dim

    def generate(self, para, debug=False) -> Task:
        world = World(self.bot_pos, self.safezone_pos, debug)
        return Task(world, self.bot_pos, para)


if __name__ == '__main__':
    def test():
        import numpy as np
        generator = TaskGenerator()
        para = np.random.random(generator.get_dim())
        generator.generate(para)


    test()
