from ..settings import Settings
from ..parameters import Parameters

from ..task_generator import TaskGenerator


def run(settings: Settings, parameters: Parameters):
    task = TaskGenerator(settings).generate(parameters, debug=False)

