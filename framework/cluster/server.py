from dataclasses import dataclass
import uuid
from typing import Optional

from icecream import ic

from ..prelude import TaskContent, StateContent, ResultContent, ClusterPacket, ClusterPacketType
from ._network import Head, ConnectionManager


@dataclass
class ClientState:
    """Represents the state of a connected client"""
    working: bool
    gpu_usage: float
    task: Optional[TaskContent] = None


class Server:
    """Minimal server for task distribution and heartbeat monitoring"""

    def __init__(self, heartbeat_interval: float = 5.0, heartbeat_timeout: float = 30.0):
        self._head = Head()
        self._connection_manager = ConnectionManager(interval=heartbeat_interval, timeout=heartbeat_timeout)
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
        dead_ids = await self._connection_manager.manage(self._head)
        for dead_id in dead_ids:
            self._client_states.pop(ic(dead_id), None)

        return {
            client_id: state
            for client_id, state in self._client_states.items()
            if client_id in self._connection_manager.get_ids()
        }

    async def get_available_clients(self) -> dict[uuid.UUID, ClientState]:
        """Get clients that are alive and not working"""
        alive_clients = await self.get_alive_clients()
        result = {}
        for client_id, state in alive_clients.items():
            if not ic(state.working) and ic(state.task) is None:
                result[client_id] = state
        return result

    def _update_client_states(self, packets: dict[uuid.UUID, ClusterPacket]) -> dict[uuid.UUID, ClusterPacket]:
        """Update client states from received packets"""

        pass_through: dict[uuid.UUID, ClusterPacket] = {}

        for i, p in packets.items():
            if p.type == ClusterPacketType.STATE and isinstance(p.content, StateContent):
                if i not in self._client_states:
                    self._client_states[i] = ClientState(
                        working=p.content.working,
                        gpu_usage=p.content.gpu_usage,
                        task=None
                    )

            elif p.type == ClusterPacketType.RESULT and isinstance(p.content, ResultContent):
                self._client_states[i].task = None
                pass_through[i] = p

            else:
                pass_through[i] = p

        return pass_through

    async def manage(self) -> set[uuid.UUID]:
        dead_ids = await self._connection_manager.manage(self._head)
        for dead_id in dead_ids:
            self._client_states.pop(dead_id, None)
        return dead_ids

    def receive(self) -> dict[uuid.UUID, ClusterPacket]:
        """Get results from all clients"""
        packets: dict[uuid.UUID, ClusterPacket] = self._connection_manager.receive(self._head)
        packets: dict[uuid.UUID, ClusterPacket] = self._update_client_states(packets)
        return packets

    async def send_task(self, client_id: uuid.UUID, task_content: TaskContent) -> bool:
        """Send a task to a specific client"""
        if client_id not in self._head.get_ids():
            print(f"Client {client_id} not found")
            return False
        elif client_id not in self._client_states:
            return False
        elif self._client_states[client_id].working:
            return False
        elif self._client_states[client_id].task is not None:
            return False

        self._client_states[client_id].task = task_content

        cluster_packet = ClusterPacket(ClusterPacketType.TASK, task_content)
        await self._head.send(client_id, cluster_packet)
        return True
