import datetime
import enum
import pickle
from typing import Self


class ClusterPacketType(enum.Enum):
    TASK = 1
    RESULT = 2
    STATE = 3


class ClusterPacket:
    def __init__(self, type_: ClusterPacketType, content):
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
