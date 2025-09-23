import asyncio
import datetime
import uuid
from typing import Optional

from framework.cluster._network._contents import PingContent
from framework.cluster._network._packets import AsyncTaskPacketType, AsyncTaskPacket


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

    def empty(self, id_: uuid.UUID = None) -> bool | dict[uuid.UUID, bool]:
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
