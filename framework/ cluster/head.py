import asyncio
from functools import partial

from .packets import AsyncTaskPacketType, AsyncTaskPacket, AsyncTaskTunnel, AsyncTaskTunnelChild, WorkerPacket


async def _send_payload(writer: asyncio.StreamWriter, payload: WorkerPacket):
    payload_size = payload.size()
    if payload_size < 4:
        raise ValueError("Payload size must be at least 4 bytes.")
    writer.write(payload_size.to_bytes(4, byteorder='big'))
    await writer.drain()

    writer.write(payload.as_bytes())
    await writer.drain()


async def _receive_payload(reader: asyncio.StreamReader, timeout: float) -> WorkerPacket:
    size_data = await asyncio.wait_for(reader.readexactly(4), timeout=timeout)
    payload_size = int.from_bytes(size_data, byteorder='big')
    if payload_size < 4:
        raise ValueError("Payload size must be at least 4 bytes.")

    payload_data = await asyncio.wait_for(reader.readexactly(payload_size), timeout=timeout)
    return WorkerPacket.from_bytes(payload_data)


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
