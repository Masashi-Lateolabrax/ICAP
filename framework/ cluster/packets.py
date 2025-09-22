import asyncio
import enum


class AsyncTaskPacketType(enum.Enum):
    STOP = 1


class AsyncTaskPacket:
    def __init__(self, type_: AsyncTaskPacketType, content):
        self.type: AsyncTaskPacketType = type_
        self.content = content

    @classmethod
    def stop_packet(cls):
        return cls(AsyncTaskPacketType.STOP, None)


class AsyncTaskTunnelChild:
    def __init__(self, parent_queue: asyncio.Queue, child_queue: asyncio.Queue):
        self.parent_queue = parent_queue
        self.child_queue = child_queue

    async def send(self, packet: AsyncTaskPacket):
        if not isinstance(packet, AsyncTaskPacket):
            raise TypeError("packet must be an instance of AsyncTaskPacket")
        await self.child_queue.put(packet)

    async def receive(self) -> AsyncTaskPacket:
        data = await self.parent_queue.get()
        if not isinstance(data, AsyncTaskPacket):
            raise TypeError("data must be an instance of AsyncTaskPacket")
        return data


class AsyncTaskTunnel:
    def __init__(self):
        self.parent_queue = asyncio.Queue()
        self.child_queue = asyncio.Queue()

    def spawn_child(self):
        return AsyncTaskTunnelChild(self.child_queue, self.parent_queue)

    async def send(self, packet: AsyncTaskPacket):
        if not isinstance(packet, AsyncTaskPacket):
            raise TypeError("packet must be an instance of AsyncTaskPacket")
        await self.parent_queue.put(packet)

    async def receive(self) -> AsyncTaskPacket:
        data = await self.child_queue.get()
        if not isinstance(data, AsyncTaskPacket):
            raise TypeError("data must be an instance of AsyncTaskPacket")
        return data
