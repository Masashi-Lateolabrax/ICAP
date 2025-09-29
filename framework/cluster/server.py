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

    def get_available_clients(self) -> set[uuid.UUID]:
        """Get clients that are alive and not working"""
        return set(i for i in self._head.get_ids() if self.is_ready(i))

    def _update_client_states(
            self, packets: dict[uuid.UUID, list[ClusterPacket]]
    ) -> dict[uuid.UUID, list[ClusterPacket]]:
        """Update client states from received packets"""

        pass_through: dict[uuid.UUID, list[ClusterPacket]] = {}

        for i, packet_list in packets.items():
            if i not in self._client_states:
                self._client_states[i] = ClientState(working=False, gpu_usage=0.0, task=None)

            pass_through[i] = []
            for p in packet_list:
                if p.type == ClusterPacketType.STATE and isinstance(p.content, StateContent):
                    self._client_states[i].working = p.content.working
                    self._client_states[i].gpu_usage = p.content.gpu_usage

                elif p.type == ClusterPacketType.RESULT and isinstance(p.content, ResultContent):
                    self._client_states[i].task = None
                    pass_through[i].append(p)

                else:
                    pass_through[i].append(p)

        return pass_through

    async def manage(self) -> dict[uuid.UUID, Optional[TaskContent]]:
        await self._connection_manager.manage(self._head)

        res = {}
        for i in self._connection_manager.get_ids():
            dead_client_state = self._client_states.pop(i, None)
            res[i] = dead_client_state.task

        return res

    def receive(self) -> dict[uuid.UUID, list[ClusterPacket]]:
        """Get results from all clients"""
        packets: dict[uuid.UUID, list[ClusterPacket]] = self._connection_manager.receive(self._head)
        packets: dict[uuid.UUID, list[ClusterPacket]] = self._update_client_states(packets)
        return packets

    def is_ready(self, client_id: uuid.UUID) -> bool:
        if client_id not in self._head.get_ids():
            return False
        elif client_id not in self._client_states:
            return False
        elif self._client_states[client_id].working:
            return False
        elif self._client_states[client_id].task is not None:
            return False
        return True

    async def send_task(self, client_id: uuid.UUID, task_content: TaskContent):
        """Send a task to a specific client"""
        if client_id not in self._head.get_ids():
            raise ValueError("Client ID not found")
        elif client_id not in self._client_states:
            raise ValueError("Client state not found")
        elif self._client_states[client_id].working:
            raise ValueError("Client is currently working")
        elif self._client_states[client_id].task is not None:
            raise ValueError("Client already has a task assigned")

        self._client_states[client_id].task = task_content

        cluster_packet = ClusterPacket(ClusterPacketType.TASK, task_content)
        await self._head.send(client_id, cluster_packet)
