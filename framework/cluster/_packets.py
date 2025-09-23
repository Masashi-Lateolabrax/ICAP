import asyncio
import dataclasses
import enum
import pickle
import uuid
from typing import Optional

import numpy as np


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
        return cls(WorkerPacketType.STATE, StatePacket(gpu_usage, working))

    def as_bytes(self) -> bytes:
        type_bytes = self.type.value.to_bytes(4, 'big')
        content_bytes = pickle.dumps(self.content) if self.content is not None else b''
        return type_bytes + content_bytes

    @classmethod
    def result_packet(cls, result: list[tuple[np.ndarray, float]]):
        return cls(WorkerPacketType.RESULT, ResultPacket(result))


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
    def ping_packet(cls, content=None):
        return cls(AsyncTaskPacketType.PING, content)


class AsyncTaskTunnelChild:
    def __init__(self, id_: uuid.UUID, parent_queue: asyncio.Queue, child_queue: asyncio.Queue):
        self.uuid = id_
        self.parent_queue = parent_queue
        self.child_queue = child_queue

    async def send(self, packet: AsyncTaskPacket):
        if not isinstance(packet, AsyncTaskPacket):
            raise TypeError("packet must be an instance of AsyncTaskPacket")
        await self.child_queue.put(packet)

    async def receive(self, timeout: float = None) -> AsyncTaskPacket:
        try:
            data = await asyncio.wait_for(self.parent_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            data = AsyncTaskPacket.timeout_packet()

        if not isinstance(data, AsyncTaskPacket):
            raise TypeError("data must be an instance of AsyncTaskPacket")

        if data.type == AsyncTaskPacketType.PING:
            # Automatically respond to roll call
            await self.send(AsyncTaskPacket.ping_packet(self.uuid))
            return await self.receive(timeout)

        return data


class AsyncTaskTunnel:
    def __init__(self):
        self.parent_queue = {}
        self.child_queue = {}
        self._buf_child_queue: dict[uuid.UUID, list[AsyncTaskPacket]] = {}

    def spawn_child(self):
        id_ = uuid.uuid4()
        parent_queue = asyncio.Queue()
        child_queue = asyncio.Queue()
        self.parent_queue[id_] = parent_queue
        self.child_queue[id_] = child_queue
        return AsyncTaskTunnelChild(id_, child_queue, parent_queue)

    async def send(self, id_: uuid.UUID, packet: AsyncTaskPacket):
        if not isinstance(packet, AsyncTaskPacket):
            raise TypeError("packet must be an instance of AsyncTaskPacket")
        await self.child_queue[id_].put(packet)

    async def receive(self, id_: uuid.UUID, timeout: float = None) -> Optional[AsyncTaskPacket]:
        queue = self.parent_queue.get(id_)
        try:
            data = await asyncio.wait_for(queue, timeout=timeout)
        except asyncio.TimeoutError:
            return None

        if not isinstance(data, AsyncTaskPacket):
            raise TypeError("data must be an instance of AsyncTaskPacket")

        return data

    async def send_and_receive(
            self, id_: uuid.UUID, packet: AsyncTaskPacket, timeout: float = None,
            expect_type: AsyncTaskPacketType = None
    ) -> Optional[AsyncTaskPacket]:
        await self.send(id_, packet)
        return await self.receive(id_, timeout, expect_type)

    def get_ids(self) -> list[uuid.UUID]:
        return list(self.child_queue.keys())

    def del_id(self, id_: uuid.UUID):
        del self.child_queue[id_]
        del self.parent_queue[id_]
        del self._buf_child_queue[id_]

    async def send_ping(self, timeout: float) -> list[uuid.UUID]:
        response = []

        for id_ in self.child_queue.keys():
            if await self._send_ping(id_, timeout):
                response.append(id_)
            else:
                del self.parent_queue[id_]
                del self.child_queue[id_]
                del self._buf_child_queue[id_]

        return response


class TaskPacket:
    def __init__(self, parameter: np.ndarray):
        self.parameter = parameter


@dataclasses.dataclass
class StatePacket:
    gpu_usage: float
    working: bool


@dataclasses.dataclass
class ResultPacket:
    result: list[tuple[np.ndarray, float]]
    rejected: bool = False
