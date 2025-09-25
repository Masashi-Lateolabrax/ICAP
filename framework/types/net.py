import dataclasses
import datetime
import enum
import pickle
from typing import Self, Optional

import numpy as np


class Content:
    pass


@dataclasses.dataclass
class TaskContent(Content):
    parameter: np.ndarray


@dataclasses.dataclass
class StateContent(Content):
    gpu_usage: float
    working: bool


@dataclasses.dataclass
class ResultContent(Content):
    result: list[tuple[np.ndarray, float]]
    start_time: Optional[datetime.datetime]
    end_time: Optional[datetime.datetime]
    rejected: Optional[TaskContent] = None

    @classmethod
    def create_rejected(cls, task: TaskContent) -> Self:
        return cls(result=[], start_time=None, end_time=None, rejected=task)


class HeartbeatContent(Content):
    def __init__(self):
        self.timestamp = datetime.datetime.now(tz=datetime.UTC)


class ClusterPacketType(enum.Enum):
    TASK = 1
    RESULT = 2
    STATE = 3
    HEARTBEAT = 4


class ClusterPacket:
    def __init__(self, type_: ClusterPacketType, content: Content):
        self.type: ClusterPacketType = type_
        self.content = content
        self.timestamp = datetime.datetime.now(tz=datetime.UTC)

    def __repr__(self):
        return f"<ClusterPacket type={self.type} content={self.content}>"

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        if len(data) < 4:
            raise ValueError("data must be at least 4 bytes long")
        type_value = int.from_bytes(data[:4], 'big')
        try:
            type_ = ClusterPacketType(type_value)
        except ValueError:
            raise ValueError(f"Invalid WorkerPacketType value: {type_value}")
        content = pickle.loads(data[4:]) if len(data) > 4 else None
        return cls(type_, content)

    def as_bytes(self) -> bytes:
        type_bytes = self.type.value.to_bytes(4, 'big')
        content_bytes = pickle.dumps(self.content) if self.content is not None else b''
        return type_bytes + content_bytes


class CoroutinePacketType(enum.Enum):
    STOP = 1
    CLUSTER_PACKET = 2


class CoroutinePacket:
    def __init__(self, type_: CoroutinePacketType, content):
        self.type: CoroutinePacketType = type_
        self.content = content

    def __repr__(self):
        return f"<AsyncTaskPacket type={self.type.name} content={self.content}>"

    @classmethod
    def stop_packet(cls):
        return cls(CoroutinePacketType.STOP, None)

    @classmethod
    def worker_packet(cls, content: ClusterPacket):
        return cls(CoroutinePacketType.CLUSTER_PACKET, content)
