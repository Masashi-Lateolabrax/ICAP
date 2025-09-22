import asyncio
from functools import partial

from .packets import (
    AsyncTaskPacketType, AsyncTaskPacket, AsyncTaskTunnel, AsyncTaskTunnelChild,
    WorkerPacket, StatePacket
)


async def _send_payload(writer: asyncio.StreamWriter, payload: WorkerPacket):
    payload_bytes = payload.as_bytes()
    payload_size = len(payload_bytes)
    if payload_size < 4:
        raise ValueError("Payload size must be at least 4 bytes.")

    writer.write(payload_size.to_bytes(4, byteorder='big'))
    writer.write(payload_bytes)
    await writer.drain()


async def _receive_payload(reader: asyncio.StreamReader, timeout: float) -> WorkerPacket:
    size_data = await asyncio.wait_for(reader.readexactly(4), timeout=timeout)
    payload_size = int.from_bytes(size_data, byteorder='big')
    if payload_size < 4:
        raise ValueError("Payload size must be at least 4 bytes.")

    payload_data = await asyncio.wait_for(reader.readexactly(payload_size), timeout=timeout)
    return WorkerPacket.from_bytes(payload_data)


async def head_routine(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter, tunnel: AsyncTaskTunnelChild, timeout: float
):
    while True:
        packet: AsyncTaskPacket = await tunnel.receive()

        if packet.type == AsyncTaskPacketType.STOP:
            break

        if packet.type == AsyncTaskPacketType.WORKER_PACKET:
            payload = packet.content
            if not isinstance(payload, WorkerPacket):
                raise ValueError("Invalid packet content type. Expected WorkerPacket for PAYLOAD type.")
            await _send_payload(writer, payload)

        try:
            response = await _receive_payload(reader, timeout)
        except asyncio.TimeoutError:
            continue

        if not isinstance(response, WorkerPacket):
            raise ValueError("Invalid response type. Expected WorkerPacket.")
        response_packet = AsyncTaskPacket(AsyncTaskPacketType.WORKER_PACKET, response)
        await tunnel.send(response_packet)


class Head:
    def __init__(self):
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
        for id_ in self.tunnel.get_ids():
            await self.tunnel.send(id_, AsyncTaskPacket.stop_packet())

    async def cleanup(self, timeout: float = 5):
        await self.tunnel.send_ping(timeout)

    async def get_ids(self):
        return self.tunnel.get_ids()

    async def _send_and_receive_worker_packet(
            self, id_, worker_packet: WorkerPacket, timeout: float
    ) -> WorkerPacket:
        packet = AsyncTaskPacket(AsyncTaskPacketType.WORKER_PACKET, worker_packet)

        response = await self.tunnel.send_and_receive(id_, packet, timeout, AsyncTaskPacketType.WORKER_PACKET)
        if not isinstance(response.content, WorkerPacket):
            raise ValueError("Invalid response type. Expected WorkerPacket.")

        return response.content

    async def get_worker_load(self, id_, num_payloads: int, timeout: float) -> float:
        packet = AsyncTaskPacket(
            AsyncTaskPacketType.WORKER_PACKET,
            WorkerPacket.request_load(num_payloads)
        )

        response = await self.tunnel.send_and_receive(id_, packet, timeout, AsyncTaskPacketType.WORKER_PACKET)
        if not isinstance(response.content, WorkerPacket):
            raise ValueError("Invalid response type. Expected WorkerPacket.")

        if not isinstance(response.content.content, float):
            raise ValueError("Invalid response type. Expected float.")

        return response.content.content
