import asyncio

from .packets import AsyncTaskPacketType, AsyncTaskPacket, AsyncTaskTunnel, AsyncTaskTunnelChild
from .woker import Worker


async def head_server(address, port, tunnel: AsyncTaskTunnelChild):
    while True:
        packet: AsyncTaskPacket = await tunnel.receive()

        if packet.type == AsyncTaskPacketType.STOP:
            break


class Head:
    def __init__(self, address, port):
        self.workers: list[Worker] = []
        self.tunnel = AsyncTaskTunnel()
        self.task = asyncio.create_task(head_server(address, port, self.tunnel.spawn_child()))

    async def stop(self):
        await self.tunnel.send(AsyncTaskPacket.stop_packet())
