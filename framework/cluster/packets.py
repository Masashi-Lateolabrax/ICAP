import asyncio
import enum
import pickle
import uuid
from typing import Optional


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

    def as_bytes(self) -> bytes:
        type_bytes = self.type.value.to_bytes(4, 'big')
        content_bytes = pickle.dumps(self.content) if self.content is not None else b''
        return type_bytes + content_bytes


class AsyncTaskPacketType(enum.Enum):
    TIMEOUT = -1
    STOP = 1
    WORKER_PACKET = 2
    PING = 3


class _AsyncTaskPacket:
    def __init__(self, type_: AsyncTaskPacketType, content):
        self.type: AsyncTaskPacketType = type_
        self.content = content

    @classmethod
    def timeout_packet(cls):
        return cls(AsyncTaskPacketType.TIMEOUT, None)

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

    async def send(self, packet: _AsyncTaskPacket):
        if not isinstance(packet, _AsyncTaskPacket):
            raise TypeError("packet must be an instance of AsyncTaskPacket")
        await self.child_queue.put(packet)

    async def receive(self, timeout: float = None) -> _AsyncTaskPacket:
        try:
            data = await asyncio.wait_for(self.parent_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            data = _AsyncTaskPacket.timeout_packet()

        if not isinstance(data, _AsyncTaskPacket):
            raise TypeError("data must be an instance of AsyncTaskPacket")

        if data.type == AsyncTaskPacketType.PING:
            # Automatically respond to roll call
            await self.send(_AsyncTaskPacket.ping_packet(self.uuid))
            return await self.receive(timeout)

        return data


class AsyncTaskTunnel:
    def __init__(self):
        self.parent_queue = {}
        self.child_queue = {}
        self._buf_child_queue: dict[uuid.UUID, list[_AsyncTaskPacket]] = {}

    def spawn_child(self):
        id_ = uuid.uuid4()
        parent_queue = asyncio.Queue()
        child_queue = asyncio.Queue()
        self.parent_queue[id_] = parent_queue
        self.child_queue[id_] = child_queue
        self._buf_child_queue[id_] = []
        return AsyncTaskTunnelChild(id_, child_queue, parent_queue)

    async def send(self, id_: uuid.UUID, packet: _AsyncTaskPacket):
        if not isinstance(packet, _AsyncTaskPacket):
            raise TypeError("packet must be an instance of AsyncTaskPacket")
        await self.child_queue[id_].put(packet)

    async def receive(
            self, id_: uuid.UUID, timeout: float = None, expect_type: AsyncTaskPacketType = None
    ) -> Optional[_AsyncTaskPacket]:
        if len(self._buf_child_queue[id_]) > 0:
            data = self._buf_child_queue[id_].pop(0)

        else:
            try:
                data = await asyncio.wait_for(self.parent_queue[id_].get(), timeout=timeout)
            except asyncio.TimeoutError:
                return _AsyncTaskPacket.timeout_packet()

            if not isinstance(data, _AsyncTaskPacket):
                raise TypeError("data must be an instance of AsyncTaskPacket")

            while expect_type is not None and data.type != expect_type:
                self._buf_child_queue[id_].append(data)

                data = None
                try:
                    data = await asyncio.wait_for(self.parent_queue[id_].get(), timeout=timeout)
                except asyncio.TimeoutError:
                    break

                if not isinstance(data, _AsyncTaskPacket):
                    raise TypeError("data must be an instance of AsyncTaskPacket")

        return data

    async def send_and_receive(
            self, id_: uuid.UUID, packet: _AsyncTaskPacket, timeout: float = None
    ) -> _AsyncTaskPacket:
        await self.send(id_, packet)
        return await self.receive(id_, timeout)

    def get_ids(self) -> list[uuid.UUID]:
        return list(self.child_queue.keys())

    async def _send_ping(self, id_: uuid.UUID, timeout: float) -> bool:
        await self.child_queue[id_].put(_AsyncTaskPacket.ping_packet())

        while True:
            try:
                data = await asyncio.wait_for(self.parent_queue[id_].get(), timeout=timeout)

                if not isinstance(data, _AsyncTaskPacket):
                    raise TypeError("data must be an instance of AsyncTaskPacket")

                if data.type == AsyncTaskPacketType.PING:
                    if data.content == id_:
                        return True

                self._buf_child_queue[id_].append(data)

            except asyncio.TimeoutError:
                return False

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
