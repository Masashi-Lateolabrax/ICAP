import src.optimizer as opt
from .task import Task


class TaskGenerator(opt.TaskGenerator):
    def generate(self, para) -> Task:
        return Task()
