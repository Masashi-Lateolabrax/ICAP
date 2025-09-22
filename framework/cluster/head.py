import asyncio
from functools import partial
from typing import Optional

from .packets import (
    AsyncTaskPacketType, AsyncTaskPacket, AsyncTaskTunnel, AsyncTaskTunnelChild,
    WorkerPacket, StatePacket
)
from .utils import receive_payload, send_payload


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
            await send_payload(writer, payload)

        try:
            response = await receive_payload(reader, timeout)
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

    async def start(self, address, port, timeout: float):
        if self.server is not None:
            raise RuntimeError("Server is already running")

        def body_fn():
            child = self.tunnel.spawn_child()
            return partial(head_routine, tunnel=child, timeout=timeout)

        self.server = await asyncio.start_server(body_fn(), address, port)

    async def stop(self):
        for id_ in self.tunnel.get_ids():
            await self.tunnel.send(id_, AsyncTaskPacket.stop_packet())

    async def cleanup(self, timeout: float = 5):
        await self.tunnel.send_ping(timeout)

    async def get_ids(self):
        return self.tunnel.get_ids()

    async def _send_and_receive_worker_packet(
            self, id_, worker_packet: WorkerPacket, timeout: float
    ) -> Optional[WorkerPacket]:
        packet = AsyncTaskPacket(AsyncTaskPacketType.WORKER_PACKET, worker_packet)

        response = await self.tunnel.send_and_receive(id_, packet, timeout, AsyncTaskPacketType.WORKER_PACKET)

        if response is None:
            return None
        if response.is_timeout():
            return WorkerPacket.timeout_packet()
        if not isinstance(response.content, WorkerPacket):
            raise ValueError("Invalid response type. Expected WorkerPacket.")

        return response.content

    async def get_worker_load(self, id_, num_payloads: int, timeout: float) -> float:
        packet = WorkerPacket.request_load(num_payloads)
        response = await self._send_and_receive_worker_packet(id_, packet, timeout)

        if response is None or response.is_timeout():
            return float("inf")
        if not isinstance(response.content.content, float):
            raise ValueError("Invalid response type. Expected float.")

        return response.content.content

    async def get_worker_state(self, id_, timeout: float) -> Optional[StatePacket]:
        packet = WorkerPacket.state_packet()
        response = await self._send_and_receive_worker_packet(id_, packet, timeout)

        if response is None or response.is_timeout():
            return None
        if not isinstance(response.content.content, StatePacket):
            raise ValueError("Invalid response type. Expected StatePacket.")

        return response.content.content
