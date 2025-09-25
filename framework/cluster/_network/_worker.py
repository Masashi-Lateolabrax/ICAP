import asyncio
from typing import Optional

from icecream import ic

from ...prelude import *
from ._tunnel import Tunnel
from ._utils import relay_routine


async def worker_routine(address: str, port: int, timeout: float, tunnel: Tunnel):
    connection = await asyncio.open_connection(address, port)
    reader: asyncio.StreamReader = connection[0]
    writer: asyncio.StreamWriter = connection[1]

    ic(writer.get_extra_info('sockname'))

    while await relay_routine(reader, writer, tunnel, timeout):
        pass


class WorkerClient:
    @classmethod
    async def start(cls, address: str, port: int, timeout: float):
        parent, child = Tunnel.create()
        routine_handler = asyncio.create_task(
            worker_routine(address, port, timeout, child)
        )
        return cls(parent, routine_handler)

    def __init__(self, tunnel: Tunnel, routine_handler: asyncio.Task):
        self.tunnel = tunnel
        self.routine_handler = routine_handler

    async def stop(self):
        await self.tunnel.send(CoroutinePacket.stop_packet())
        await self.routine_handler
        self.routine_handler = None
        self.tunnel = None

    def receive(self) -> Optional[CoroutinePacket]:
        return self.tunnel.receive()

    async def send(self, packet: CoroutinePacket):
        await self.tunnel.send(packet)
