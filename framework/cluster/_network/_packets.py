import datetime
import enum
import pickle
import uuid

import numpy as np

from ._contents import StateContent, PingContent, ResultContent


class WorkerPacketType(enum.Enum):
    TASK = 1
    RESULT = 2
    STATE = 3


class WorkerPacket:
    def __init__(self, type_: WorkerPacketType, content, expiry: float = 1):
        self.type: WorkerPacketType = type_
        self.content = content
        self.timestamp = datetime.datetime.now(tz=datetime.UTC)
        self.expiry = expiry

    def __repr__(self):
        life = (self.timestamp + datetime.timedelta(seconds=self.expiry)) - datetime.datetime.now(tz=datetime.UTC)
        return f"<WorkerPacket type={self.type.name} content={self.content} lifetime={life.total_seconds():.2f}s>"

    @classmethod
    def from_bytes(cls, data: bytes) -> "WorkerPacket":
        if len(data) < 4:
            raise ValueError("data must be at least 4 bytes long")
        type_value = int.from_bytes(data[:4], 'big')
        try:
            type_ = WorkerPacketType(type_value)
        except ValueError:
            raise ValueError(f"Invalid WorkerPacketType value: {type_value}")
        content = pickle.loads(data[4:]) if len(data) > 4 else None
        return cls(type_, content)

    @classmethod
    def request_state(cls):
        return cls(WorkerPacketType.STATE, None)

    @classmethod
    def state_packet(cls, gpu_usage: float, working: bool):
        return cls(WorkerPacketType.STATE, StateContent(gpu_usage, working))

    def as_bytes(self) -> bytes:
        type_bytes = self.type.value.to_bytes(4, 'big')
        content_bytes = pickle.dumps(self.content) if self.content is not None else b''
        return type_bytes + content_bytes

    @classmethod
    def result_packet(
            cls, result: list[tuple[np.ndarray, float]], start_time: datetime.datetime, end_time: datetime.datetime
    ):
        return cls(WorkerPacketType.RESULT, ResultContent(result, start_time, end_time))

    @classmethod
    def reject_packet(cls):
        return cls(WorkerPacketType.RESULT, ResultContent([], None, None, rejected=True))


class AsyncTaskPacketType(enum.Enum):
    STOP = 1
    WORKER_PACKET = 2
    PING = 3


class AsyncTaskPacket:
    def __init__(self, type_: AsyncTaskPacketType, content):
        self.type: AsyncTaskPacketType = type_
        self.content = content

    def __repr__(self):
        return f"<AsyncTaskPacket type={self.type.name} content={self.content}>"

    @classmethod
    def stop_packet(cls):
        return cls(AsyncTaskPacketType.STOP, None)

    @classmethod
    def worker_packet(cls, content: WorkerPacket):
        return cls(AsyncTaskPacketType.WORKER_PACKET, content)

    @classmethod
    def ping_packet(cls, id_: uuid.UUID):
        content = PingContent(id_)
        return cls(AsyncTaskPacketType.PING, content)
