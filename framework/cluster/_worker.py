import asyncio
from typing import Optional

from icecream import ic

from ._network import (
    SingleAsyncTaskTunnel, AsyncTaskTunnelChild,
    AsyncTaskPacketType, AsyncTaskPacket, WorkerPacketType, WorkerPacket,
    ResultContent,
)
from ._utils import relay_routine


async def worker_routine(address: str, port: int, timeout: float, tunnel: AsyncTaskTunnelChild):
    connection = await asyncio.open_connection(address, port)
    reader: asyncio.StreamReader = connection[0]
    writer: asyncio.StreamWriter = connection[1]

    ic(writer.get_extra_info('sockname'))

    while await relay_routine(reader, writer, tunnel, timeout):
        pass


class WorkerClient:
    def __init__(self):
        self.routine_handler: Optional[asyncio.Task] = None
        self.tunnel = SingleAsyncTaskTunnel()
        self.buffer: list[WorkerPacket] = []

    async def start(self, address: str, port: int, timeout: float):
        if self.routine_handler is not None:
            raise

        self.routine_handler = asyncio.create_task(
            worker_routine(address, port, timeout, self.tunnel.spawn_child())
        )

    async def stop(self):
        await self.tunnel.send(AsyncTaskPacket.stop_packet())

    def _cleanup_buffer(self):
        latest_state_packet = {
            "index": None,
            "packet": None
        }
        for i in reversed(range(len(self.buffer))):
            packet = self.buffer[i]
            if packet.type != WorkerPacketType.STATE:
                continue
            if latest_state_packet["packet"] is None:
                latest_state_packet["index"] = i
                latest_state_packet["packet"] = packet
                continue
            elif latest_state_packet["packet"].timestamp < packet.timestamp:
                latest_state_packet["packet"] = packet
            self.buffer.pop(latest_state_packet["index"])
            latest_state_packet["index"] = i

        if latest_state_packet["index"] is not None:
            self.buffer[latest_state_packet["index"]] = latest_state_packet["packet"]

    async def receive(self) -> Optional[WorkerPacket]:
        while not ic(self.tunnel.empty()):
            response = await self.tunnel.receive()
            if response is None:
                raise RuntimeError("There is not reachable")
            if response.type != AsyncTaskPacketType.WORKER_PACKET:
                raise ValueError("Invalid packet type. Expected WORKER_PACKET.")
            if not isinstance(response.content, WorkerPacket):
                raise ValueError("Invalid packet content. Expected WorkerPacket.")
            self.buffer.append(response.content)

        if len(self.buffer) == 0:
            return None

        self._cleanup_buffer()
        return self.buffer.pop(0)

    async def send_worker_state(self, gpu_usage: float, working: bool):
        packet = WorkerPacket.state_packet(gpu_usage, working)
        packet = AsyncTaskPacket.worker_packet(packet)
        await self.tunnel.send(packet)

    async def send_worker_result(self, result: ResultContent):
        packet = WorkerPacket(WorkerPacketType.RESULT, result)
        packet = AsyncTaskPacket.worker_packet(packet)
        await self.tunnel.send(packet)
