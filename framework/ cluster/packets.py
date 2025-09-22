import asyncio
import enum
import pickle
import uuid


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
        if len(self._buf_child_queue[id_]) > 0:
            data = self._buf_child_queue[id_].pop(0)
        else:
            try:
                data = await asyncio.wait_for(self.parent_queue[id_].get(), timeout=timeout)
            except asyncio.TimeoutError:
                data = AsyncTaskPacket.timeout_packet()
        if not isinstance(data, AsyncTaskPacket):
            raise TypeError("data must be an instance of AsyncTaskPacket")
        return data

    def get_ids(self) -> list[uuid.UUID]:
        return list(self.child_queue.keys())

    async def _send_roll_call_packet(self, id_: uuid.UUID, timeout: float) -> bool:
        await self.child_queue[id_].put(AsyncTaskPacket.roll_call_packet())

        while True:
            try:
                data = await asyncio.wait_for(self.parent_queue[id_].get(), timeout=timeout)

                if not isinstance(data, AsyncTaskPacket):
                    raise TypeError("data must be an instance of AsyncTaskPacket")

                if data.type == AsyncTaskPacketType.ROLL_CALL:
                    if data.content == id_:
                        return True

                self._buf_child_queue[id_].append(data)

            except asyncio.TimeoutError:
                return False

    async def calling_roll(self, timeout: float) -> list[uuid.UUID]:
        response = []

        for id_ in self.child_queue.keys():
            if await self._send_roll_call_packet(id_, timeout):
                response.append(id_)
            else:
                del self.parent_queue[id_]
                del self.child_queue[id_]
                del self._buf_child_queue[id_]

        return response
