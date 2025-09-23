import asyncio
import datetime
import enum
import pickle
import uuid
from typing import Optional

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


class AsyncTaskTunnelChild:
    def __init__(self, id_: uuid.UUID, receiver: asyncio.Queue, sender: asyncio.Queue):
        self.uuid = id_
        self.receiver = receiver
        self.sender = sender

    async def send(self, packet: AsyncTaskPacket):
        if not isinstance(packet, AsyncTaskPacket):
            raise TypeError("packet must be an instance of AsyncTaskPacket")
        await self.sender.put(packet)

    async def empty(self) -> bool:
        return self.receiver.empty()

    async def receive(self, timeout: float = None) -> Optional[AsyncTaskPacket]:
        try:
            data = await asyncio.wait_for(self.receiver.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

        if not isinstance(data, AsyncTaskPacket):
            raise TypeError("data must be an instance of AsyncTaskPacket")

        if data.type == AsyncTaskPacketType.PING:
            # Automatically respond to roll call
            if not isinstance(data.content, PingContent):
                raise ValueError("Invalid ping packet content")
            if data.content.id != self.uuid:
                raise ValueError("Ping packet ID does not match tunnel ID")
            data.content.response_time = datetime.datetime.now(tz=datetime.UTC)
            await self.send(data)
            return await self.receive(timeout)

        return data


class AsyncTaskTunnel:
    def __init__(self):
        self.receiver = {}
        self.sender = {}
        self._buf_child_queue: dict[uuid.UUID, list[AsyncTaskPacket]] = {}

    def spawn_child(self):
        id_ = uuid.uuid4()
        receiver = asyncio.Queue()
        sender = asyncio.Queue()
        self.receiver[id_] = receiver
        self.sender[id_] = sender
        return AsyncTaskTunnelChild(id_, sender, receiver)

    async def send(self, id_: uuid.UUID, packet: AsyncTaskPacket):
        if not isinstance(packet, AsyncTaskPacket):
            raise TypeError("packet must be an instance of AsyncTaskPacket")
        await self.sender[id_].put(packet)

    async def empty(self, id_: uuid.UUID = None) -> bool | dict[uuid.UUID, bool]:
        if id_ is not None:
            return self.receiver[id_].empty()
        return {i: q.empty() for i, q in self.receiver.items()}

    async def receive(self, id_: uuid.UUID, timeout: float = None) -> Optional[AsyncTaskPacket]:
        queue = self.receiver.get(id_)
        try:
            data = await asyncio.wait_for(queue, timeout=timeout)
        except asyncio.TimeoutError:
            return None

        if not isinstance(data, AsyncTaskPacket):
            raise TypeError("data must be an instance of AsyncTaskPacket")

        return data

    def get_ids(self) -> list[uuid.UUID]:
        return list(self.sender.keys())

    def del_id(self, id_: uuid.UUID):
        del self.sender[id_]
        del self.receiver[id_]
        del self._buf_child_queue[id_]

    async def send_ping(self, id_: uuid.UUID) -> PingContent:
        packet = AsyncTaskPacket.ping_packet(id_)
        await self.send(id_, packet)
        return packet.content
