import asyncio
from functools import partial

from .packets import AsyncTaskPacketType, AsyncTaskPacket, AsyncTaskTunnel, AsyncTaskTunnelChild
from .woker import Worker


async def head_routine(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, tunnel: AsyncTaskTunnelChild):
    running = True
    while running:
        packet: AsyncTaskPacket = await tunnel.receive()

        if packet.type == AsyncTaskPacketType.STOP:
            running = False


class Head:
    def __init__(self):
        self.workers: list[Worker] = []
        self.tunnel = AsyncTaskTunnel()
        self.server = None

    async def start(self, address, port):
        if self.server is not None:
            raise RuntimeError("Server is already running")
        self.server = await asyncio.start_server(
            partial(head_routine, tunnel=self.tunnel.spawn_child()),
            address, port
        )

    async def stop(self):
        await self.tunnel.send(AsyncTaskPacket.stop_packet())
