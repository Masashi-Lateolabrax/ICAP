import os

from lib.setting import generate

generate(os.path.join(os.path.dirname(__file__), "settings.yaml"))

from .settings import Settings
from .core import Robot, RobotDebugBuf
from ._task_generator import TaskGenerator
from ._task import Task
