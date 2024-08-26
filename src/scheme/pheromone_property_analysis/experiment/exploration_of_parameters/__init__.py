import os.path

from lib.setting import generate

generate(
    os.path.join(os.path.dirname(__file__), "settings.yaml")
)

from .settings import Settings
from ._utils import convert_para
from ._task import Task, Generator
from ._rec import TaskForRec, DecTaskForRec
