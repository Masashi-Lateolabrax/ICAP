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
