import os.path

from lib.setting import generate

generate(
    os.path.join(os.path.dirname(__file__), "settings.yaml")
)

from .settings import Settings
from ._xml_setting import gen_xml
from ._data_collector import DataCollector
from ._task import AnalysisEnvironment
