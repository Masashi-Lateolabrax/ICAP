from dataclasses import dataclass
import uuid

from ..prelude import TaskContent, StateContent, ClusterPacket, ClusterPacketType, CoroutinePacket, CoroutinePacketType
from ._network import Head, ConnectionManager


@dataclass
class ClientState:
    """Represents the state of a connected client"""
    working: bool
    gpu_usage: float


class SimpleServer:
    """Minimal server for task distribution and heartbeat monitoring"""

    def __init__(self, heartbeat_timeout: float = 30.0):
        self._head = Head()
        self._connection_manager = ConnectionManager(interval=5.0, timeout=heartbeat_timeout)
        self._client_states: dict[uuid.UUID, ClientState] = {}
        self._running = False

    async def start(self, host: str, port: int, timeout: float = 5.0) -> None:
        """Start the server"""
        await self._head.start(host, port, timeout)
        self._running = True

        print(f"SimpleServer started on {host}:{port}")

    async def stop(self) -> None:
        """Stop the server"""
        self._running = False
        await self._head.stop()

    async def get_alive_clients(self) -> dict[uuid.UUID, ClientState]:
        """Get all clients that are currently alive"""
        return {
            client_id: state
            for client_id, state in self._client_states.items()
            if client_id in self._head.get_ids()
        }

    async def get_available_clients(self) -> dict[uuid.UUID, ClientState]:
        """Get clients that are alive and not working"""
        alive_clients = await self.get_alive_clients()
        return {
            client_id: state
            for client_id, state in alive_clients.items()
            if not state.working
        }

    async def send_task(self, client_id: uuid.UUID, task_content: TaskContent) -> bool:
        """Send a task to a specific client"""
        if client_id not in self._head.get_ids():
            print(f"Client {client_id} not found")
            return False

        try:
            cluster_packet = ClusterPacket(ClusterPacketType.TASK, task_content)
            coroutine_packet = CoroutinePacket(CoroutinePacketType.CLUSTER_PACKET, cluster_packet)
            await self._head.send(client_id, coroutine_packet)
            return True
        except Exception as e:
            print(f"Failed to send task to client {client_id}: {e}")
            return False

    def _update_client_states(self, packets: dict[uuid.UUID, ClusterPacket]) -> dict[uuid.UUID, ClusterPacket]:
        """Update client states from received packets"""

        pass_through: dict[uuid.UUID, ClusterPacket] = {}
        state_packets: dict[uuid.UUID, StateContent] = {}
        for i, p in packets.items():
            if p.type == ClusterPacketType.STATE and isinstance(p.content, StateContent):
                state_packets[i] = p.content
            else:
                pass_through[i] = p

        for client_id, state in state_packets.items():
            self._client_states[client_id] = ClientState(
                working=state.working,
                gpu_usage=state.gpu_usage
            )

        return pass_through

    def get_results(self) -> dict[uuid.UUID, ClusterPacket]:
        """Get results from all clients"""
        try:
            packets = self._head.receive()
            results = {}
            for client_id, packet in packets.items():
                if isinstance(packet.content, ClusterPacket):
                    results[client_id] = packet.content
            return results
        except Exception as e:
            print(f"Failed to get results: {e}")
            return {}
