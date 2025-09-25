from typing import Optional

from ..prelude import TaskContent, StateContent, ClusterPacket, ClusterPacketType, CoroutinePacket, CoroutinePacketType
from ._network import Worker


class SimpleClient:
    """Minimal client with periodic state sending and task processing"""

    @classmethod
    async def new(cls, host: str, port: int, timeout: float = 1.0):
        """Create and start a new client"""
        worker = await Worker.start(host, port, timeout)
        client = cls(worker)
        print(f"SimpleClient connected to {host}:{port}")
        return client

    def __init__(self, worker: Worker):
        self._worker: Optional[Worker] = worker
        self._current_task: Optional[TaskContent] = None

    async def stop(self):
        """Stop the client"""
        if self._worker:
            await self._worker.stop()
            self._worker = None
        print("SimpleClient disconnected")

    async def _send_state(self):
    def _receive(self) -> Optional[ClusterPacket]:
        """Receive task from server if available"""
        if not self._worker:
            return None

        packet = self._worker.receive()
        if packet is None:
            return None
        if packet.type != CoroutinePacketType.CLUSTER_PACKET:
            raise ValueError("Unexpected packet type")
        if not isinstance(packet.content, ClusterPacket):
            raise ValueError("Unexpected content type")

        return packet.content

        """Send current state to server"""
        if not self._worker:
            return

        state_content = StateContent(
            gpu_usage=0.0,  # Placeholder - could be extended to get actual GPU usage
            working=self._current_task is not None
        )
        cluster_packet = ClusterPacket(ClusterPacketType.STATE, state_content)
        coroutine_packet = CoroutinePacket(CoroutinePacketType.CLUSTER_PACKET, cluster_packet)
        await self._worker.send(coroutine_packet)
