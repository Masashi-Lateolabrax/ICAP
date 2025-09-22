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


class AsyncTaskPacket:
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
        self._buf_child_queue[id_] = []
        return AsyncTaskTunnelChild(id_, child_queue, parent_queue)

    async def send(self, id_: uuid.UUID, packet: AsyncTaskPacket):
        if not isinstance(packet, AsyncTaskPacket):
            raise TypeError("packet must be an instance of AsyncTaskPacket")
        await self.child_queue[id_].put(packet)

    async def receive(
            self, id_: uuid.UUID, timeout: float = None, expect_type: AsyncTaskPacketType = None
    ) -> Optional[AsyncTaskPacket]:
        """
        Asynchronously receive a packet from the specified task queue.

        This method implements a complex packet retrieval system that handles buffering,
        type filtering, and timeout management. It first checks buffered packets, then
        attempts to receive from the queue, optionally filtering by packet type.

        Args:
            id_ (uuid.UUID): The unique identifier of the task queue to receive from.
            timeout (float, optional): Maximum time to wait for a packet in seconds.
                If None, waits indefinitely. Defaults to None.
            expect_type (AsyncTaskPacketType, optional): Expected packet type to filter for.
                If specified, packets of other types will be buffered for later retrieval.
                Defaults to None (accept any packet type).

        Returns:
            Optional[AsyncTaskPacket]: The received packet, timeout packet if timeout occurred,
            or None if no matching packet type was found within the timeout period.

        Raises:
            TypeError: If the received data is not an instance of _AsyncTaskPacket.

        Behavior:
            1. First checks if buffered packets exist for the given ID and returns the first one
            2. If no buffered packets, waits for a new packet from the queue
            3. If expect_type is specified, continues receiving until a matching type is found
            4. Non-matching packets are buffered in _buf_child_queue for later retrieval
            5. Returns timeout packet if timeout occurs during any wait operation
        """
        if len(self._buf_child_queue[id_]) > 0:
            for i, data in enumerate(self._buf_child_queue[id_]):
                if expect_type is not None and data.type == expect_type:
                    return self._buf_child_queue[id_].pop(i)

        try:
            data = await asyncio.wait_for(self.parent_queue[id_].get(), timeout=timeout)
        except asyncio.TimeoutError:
            return AsyncTaskPacket.timeout_packet()

        if not isinstance(data, AsyncTaskPacket):
            raise TypeError("data must be an instance of AsyncTaskPacket")

        while expect_type is not None and data.type != expect_type:
            self._buf_child_queue[id_].append(data)

            data = None
            try:
                data = await asyncio.wait_for(self.parent_queue[id_].get(), timeout=timeout)
            except asyncio.TimeoutError:
                break

            if not isinstance(data, AsyncTaskPacket):
                raise TypeError("data must be an instance of AsyncTaskPacket")

        return data

    async def send_and_receive(
            self, id_: uuid.UUID, packet: AsyncTaskPacket, timeout: float = None,
            expect_type: AsyncTaskPacketType = None
    ) -> AsyncTaskPacket:
        await self.send(id_, packet)
        return await self.receive(id_, timeout, expect_type)

    def get_ids(self) -> list[uuid.UUID]:
        return list(self.child_queue.keys())

    async def _send_ping(self, id_: uuid.UUID, timeout: float) -> bool:
        response = await self.send_and_receive(id_, AsyncTaskPacket.ping_packet(), timeout, AsyncTaskPacketType.PING)
        return response is not None and response.type != AsyncTaskPacketType.TIMEOUT

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
