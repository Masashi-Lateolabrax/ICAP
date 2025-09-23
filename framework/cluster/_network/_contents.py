import dataclasses
import uuid
import datetime
from typing import Optional

import numpy as np


class TaskContent:
    def __init__(self, parameter: np.ndarray):
        self.parameter = parameter


@dataclasses.dataclass
class PingContent:
    def __init__(self, id_: uuid.UUID):
        self.id = id_
        self.create_time = datetime.datetime.now(tz=datetime.UTC)
        self.response_time: Optional[datetime.datetime] = None


@dataclasses.dataclass
class StateContent:
    gpu_usage: float
    working: bool


@dataclasses.dataclass
class ResultContent:
    result: list[tuple[np.ndarray, float]]
    start_time: Optional[datetime.datetime]
    end_time: Optional[datetime.datetime]
    rejected: bool = False
