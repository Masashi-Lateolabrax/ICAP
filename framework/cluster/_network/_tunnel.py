import asyncio
import uuid
from typing import Optional

from icecream import ic

from ...prelude import *


class Tunnel:
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
