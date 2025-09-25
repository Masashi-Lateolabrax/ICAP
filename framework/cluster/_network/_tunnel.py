import asyncio
import uuid
from typing import Optional

from icecream import ic

from ...prelude import *


class Tunnel:
    @classmethod
    def create(cls) -> tuple['Tunnel', 'Tunnel']:
        queue1: asyncio.Queue = asyncio.Queue()
        queue2: asyncio.Queue = asyncio.Queue()
        id_ = uuid.uuid4()
        return cls(id_, queue1, queue2), cls(id_, queue2, queue1)

    def __init__(self, id_: uuid.UUID, receiver: asyncio.Queue, sender: asyncio.Queue):
        self.id = id_
        self.receiver = receiver
        self.sender = sender

    async def send(self, packet: CoroutinePacket):
        if not isinstance(packet, CoroutinePacket):
            raise TypeError("packet must be an instance of AsyncTaskPacket")
        await self.sender.put(packet)

    def receive(self) -> Optional[CoroutinePacket]:
        if self.receiver.empty():
            return None
        packet = ic(self.receiver.get_nowait())
        if not isinstance(packet, CoroutinePacket):
            raise TypeError("packet must be an instance of AsyncTaskPacket")
        return packet
