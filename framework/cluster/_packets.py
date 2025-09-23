import enum
import pickle
import uuid

import numpy as np

from ._contents import StateContent, PingContent, ResultContent


class WorkerPacketType(enum.Enum):
    TASK = 1
    RESULT = 2
    STATE = 3
    LOAD = 4


class WorkerPacket:
    def __init__(self, type_: WorkerPacketType, content):
        self.type: WorkerPacketType = type_
        self.content = content

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
    def request_load(cls, num_payloads: int):
        return cls(WorkerPacketType.LOAD, num_payloads)

    @classmethod
    def load_packet(cls, load: float):
        return cls(WorkerPacketType.LOAD, load)

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
    def result_packet(cls, result: list[tuple[np.ndarray, float]]):
        return cls(WorkerPacketType.RESULT, ResultContent(result))


class AsyncTaskPacketType(enum.Enum):
    STOP = 1
    WORKER_PACKET = 2
    PING = 3


class AsyncTaskPacket:
    def __init__(self, type_: AsyncTaskPacketType, content):
        self.type: AsyncTaskPacketType = type_
        self.content = content

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
