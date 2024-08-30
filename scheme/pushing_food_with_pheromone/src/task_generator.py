import libs.optimizer as opt

from .task import Task
from .world import World


class TaskGenerator(opt.TaskGenerator):
    @staticmethod
    def get_dim():
        return World.get_dim()

    def generate(self, para, debug=False) -> Task:
        return Task(para, debug)
