from dataclasses import dataclass
import uuid
from typing import Optional

from ..prelude import TaskContent, StateContent, ResultContent, ClusterPacket, ClusterPacketType
from ._network import Head, ConnectionManager


@dataclass
class ClientState:
    """Represents the state of a connected client"""
    working: bool
    gpu_usage: float
    task: Optional[TaskContent] = None


class SimpleServer:
    """Minimal server for task distribution and heartbeat monitoring"""

    def __init__(self, heartbeat_timeout: float = 30.0):
        self._head = Head()
        self._connection_manager = ConnectionManager(interval=5.0, timeout=heartbeat_timeout)
        self._client_states: dict[uuid.UUID, ClientState] = {}

    async def start(self, host: str, port: int, timeout: float = 5.0) -> None:
        """Start the server"""
        await self._head.start(host, port, timeout)
        print(f"SimpleServer started on {host}:{port}")

    async def stop(self) -> None:
        """Stop the server"""
        await self._head.stop()

    async def get_alive_clients(self) -> dict[uuid.UUID, ClientState]:
        """Get all clients that are currently alive"""
        dead_ids = self._connection_manager.manage(self._head)
        for dead_id in dead_ids:
            self._client_states.pop(dead_id, None)

        return {
            client_id: state
            for client_id, state in self._client_states.items()
            if client_id in self._connection_manager.get_ids()
        }

    async def get_available_clients(self) -> dict[uuid.UUID, ClientState]:
        """Get clients that are alive and not working"""
        alive_clients = await self.get_alive_clients()
        return {
            client_id: state
            for client_id, state in alive_clients.items()
            if not state.working
        }

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
        self._connection_manager.manage(self._head)
        packets: dict[uuid.UUID, ClusterPacket] = self._connection_manager.receive(self._head)

        for client_id, packet in packets.items():
            if packet.type == ClusterPacketType.RESULT and isinstance(packet.content, ResultContent):
                if client_id not in self._client_states:
                    raise RuntimeError("Client {} not found".format(client_id))
                self._client_states[client_id].task = None

        packets: dict[uuid.UUID, ClusterPacket] = self._update_client_states(packets)

        return packets

    async def send_task(self, client_id: uuid.UUID, task_content: TaskContent) -> bool:
        """Send a task to a specific client"""
        self._connection_manager.manage(self._head)

        if client_id not in self._head.get_ids():
            print(f"Client {client_id} not found")
            return False
        elif client_id not in self._client_states:
            return False
        elif self._client_states[client_id].working:
            return False
        elif self._client_states[client_id].task is not None:
            return False

        current_state = self._client_states[client_id]
        current_state.task = task_content

        cluster_packet = ClusterPacket(ClusterPacketType.TASK, task_content)
        await self._head.send(client_id, cluster_packet)
        return True
