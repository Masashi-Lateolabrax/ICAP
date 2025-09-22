import asyncio
import enum
import pickle
import uuid


class AsyncTaskPacketType(enum.Enum):
    TIMEOUT = -1
    STOP = 1
    PAYLOAD = 2
    ROLL_CALL = 3


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
    def payload_packet(cls, content: "WorkerPacket"):
        return cls(AsyncTaskPacketType.PAYLOAD, content)

    @classmethod
    def roll_call_packet(cls, content=None):
        return cls(AsyncTaskPacketType.ROLL_CALL, content)


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

        if data.type == AsyncTaskPacketType.ROLL_CALL:
            # Automatically respond to roll call
            await self.send(AsyncTaskPacket.roll_call_packet(self.uuid))
            return await self.receive(timeout)

        return data


class AsyncTaskTunnel:
    def __init__(self):
        self.parent_queue = {}
        self.child_queue = {}
        self._buf_child_queue = {}

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

    async def receive(self, id_: uuid.UUID, timeout: float = None) -> AsyncTaskPacket:
        try:
            data = await asyncio.wait_for(self.parent_queue[id_].get(), timeout=timeout)
        except asyncio.TimeoutError:
            data = AsyncTaskPacket.timeout_packet()
        if not isinstance(data, AsyncTaskPacket):
            raise TypeError("data must be an instance of AsyncTaskPacket")
        return data


class WorkerPacketType(enum.Enum):
    ONEWAY = 1
    REQUEST = 2
    RETURN = 4

    TASK = 8
    RESULT = 16
    STATE = 32
    LOAD = 64


class WorkerPacket:
    def __init__(self, type_: WorkerPacketType, content):
        self.type: WorkerPacketType = type_
        self.content = content

        type_bytes = self.type.value.to_bytes(4, 'big')
        content_bytes = pickle.dumps(self.content) if self.content is not None else b''
        self.bytes = type_bytes + content_bytes

    def size(self) -> int:
        return len(self.bytes)

    def as_bytes(self) -> bytes:
        return self.bytes
