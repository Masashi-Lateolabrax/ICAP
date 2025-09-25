import asyncio
import datetime
import uuid
from typing import Optional

from icecream import ic

from framework.cluster._network._contents import PingContent
from framework.cluster._network._packets import AsyncTaskPacketType, AsyncTaskPacket


class AsyncTaskTunnelChild:
    def __init__(self, id_: uuid.UUID, receiver: asyncio.Queue, sender: asyncio.Queue):
        self.uuid = id_
        self.receiver = receiver
        self.sender = sender
        self._buffer: list[AsyncTaskPacket] = []

    async def send(self, packet: AsyncTaskPacket):
        if not isinstance(packet, AsyncTaskPacket):
            raise TypeError("packet must be an instance of AsyncTaskPacket")
        await self.sender.put(packet)

    async def update(self):
        while not self.receiver.empty():
            packet = ic(self.receiver.get_nowait())

            if not isinstance(packet, AsyncTaskPacket):
                raise TypeError("packet must be an instance of AsyncTaskPacket")

            if packet.type == AsyncTaskPacketType.PING:
                # Automatically respond to roll call
                if not isinstance(packet.content, PingContent):
                    raise ValueError("Invalid ping packet content")
                if packet.content.id != self.uuid:
                    raise ValueError("Ping packet ID does not match tunnel ID")
                packet.content.response_time = datetime.datetime.now(tz=datetime.UTC)
                await self.send(
                    AsyncTaskPacket(type_=AsyncTaskPacketType.PING, content=packet.content)
                )
                continue

            self._buffer.append(packet)

    def receive(self) -> Optional[AsyncTaskPacket]:
        return self._buffer.pop(0) if self._buffer else None


class SingleAsyncTaskTunnel:
    def __init__(self):
        self.id = uuid.uuid4()
        self.receiver = asyncio.Queue()
        self.sender = asyncio.Queue()

    def spawn_child(self):
        return AsyncTaskTunnelChild(
            id_=self.id,
            receiver=self.sender,
            sender=self.receiver
        )

    def get_id(self) -> uuid.UUID:
        return self.id

    async def send(self, packet: AsyncTaskPacket):
        if not isinstance(packet, AsyncTaskPacket):
            raise TypeError("packet must be an instance of AsyncTaskPacket")
        await self.sender.put(packet)

    def empty(self) -> bool:
        return self.receiver.empty()

    async def receive(self, timeout: float = None) -> Optional[AsyncTaskPacket]:
        try:
            data = await asyncio.wait_for(self.receiver.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

        if not isinstance(data, AsyncTaskPacket):
            raise TypeError("data must be an instance of AsyncTaskPacket")

        return data


class AsyncTaskTunnel:
    def __init__(self):
        self.tunnel: dict[uuid.UUID, SingleAsyncTaskTunnel] = {}

    def spawn_child(self):
        single = SingleAsyncTaskTunnel()
        self.tunnel[single.get_id()] = single
        return single.spawn_child()

    async def send(self, id_: uuid.UUID, packet: AsyncTaskPacket):
        if id_ not in self.tunnel:
            raise KeyError(f"No tunnel with ID {id_}")
        await self.tunnel[id_].send(packet)

    def empty(self, id_: uuid.UUID = None) -> bool | dict[uuid.UUID, bool]:
        if id_ is not None:
            return self.tunnel[id_].empty()
        return {i: q.empty() for i, q in self.tunnel.items()}

    async def receive(self, id_: uuid.UUID, timeout: float = None) -> Optional[AsyncTaskPacket]:
        return await self.tunnel[id_].receive(timeout)

    def get_ids(self) -> list[uuid.UUID]:
        return list(self.tunnel.keys())

    def del_id(self, id_: uuid.UUID):
        del self.tunnel[id_]

    async def send_ping(self, id_: uuid.UUID) -> PingContent:
        packet = AsyncTaskPacket.ping_packet(id_)
        await self.send(id_, packet)
        return packet.content
