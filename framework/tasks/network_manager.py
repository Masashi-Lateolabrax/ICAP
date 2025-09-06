import logging
import asyncio
import pickle
import struct
import socket
from typing import Optional, Dict, Set, Callable

from ..prelude import *
from .shared_task_manager import SharedTaskManager


async def _send_tasks_to_stream(tasks: dict[bytes, Task], writer: asyncio.StreamWriter) -> bool:
    try:
        data = pickle.dumps(tasks)
        size = len(data)

        # Send size header (4 bytes)
        size_header = struct.pack('!I', size)
        writer.write(size_header)

        # Send data
        writer.write(data)
        await writer.drain()

        return True
    except Exception as e:
        logging.error(f"Error sending tasks: {e}")
        return False


async def _receive_tasks_from_stream(reader: asyncio.StreamReader) -> Optional[dict[bytes, Task]]:
    try:
        # Receive size header (4 bytes)
        size_header = await reader.readexactly(4)
        if not size_header:
            return None

        size = struct.unpack('!I', size_header)[0]

        # Receive data
        data = await reader.readexactly(size)
        if not data:
            return None

        tasks = pickle.loads(data)
        return tasks

    except asyncio.IncompleteReadError:
        logging.debug("Connection closed by peer")
        return None
    except Exception as e:
        logging.error(f"Error receiving tasks: {e}")
        return None


class NetworkServer:
    def __init__(self):
        self.server: Optional[asyncio.Server] = None
        self.clients: Dict[str, asyncio.StreamWriter] = {}
        self.message_handlers: Dict[str, Callable] = {}

    async def start(self, port: int, host: str = '0.0.0.0') -> bool:
        if self.server:
            logging.warning("Server already started")
            return False

        try:
            self.server = await asyncio.start_server(
                self._handle_client, host, port
            )
            logging.info(f"TCP server started on {host}:{port}")
            return True
        except Exception as e:
            logging.error(f"Failed to start server: {e}")
            return False

    async def serve_forever(self):
        if not self.server:
            logging.error("Server not started")
            return

        async with self.server:
            await self.server.serve_forever()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        client_address = writer.get_extra_info('peername')
        client_id = f"{client_address[0]}:{client_address[1]}"

        logging.info(f"Client {client_id} connected")
        self.clients[client_id] = writer

        try:
            while True:
                tasks = await _receive_tasks_from_stream(reader)
                if tasks is None:
                    break

                # Handle received tasks
                if 'task_handler' in self.message_handlers:
                    await self.message_handlers['task_handler'](tasks, client_id)

        except Exception as e:
            logging.error(f"Error handling client {client_id}: {e}")
        finally:
            logging.info(f"Client {client_id} disconnected")
            if client_id in self.clients:
                del self.clients[client_id]
            writer.close()
            await writer.wait_closed()

    def register_handler(self, event_type: str, handler: Callable):
        self.message_handlers[event_type] = handler

    async def send_tasks_to_client(self, tasks: dict[bytes, Task], client_id: str) -> bool:
        if client_id not in self.clients:
            logging.error(f"Client {client_id} not connected")
            return False

        writer = self.clients[client_id]
        return await _send_tasks_to_stream(tasks, writer)

    async def broadcast_tasks(self, tasks: dict[bytes, Task]) -> int:
        if not self.clients:
            logging.warning("No clients connected")
            return 0

        successful_sends = 0
        failed_clients = []

        for client_id, writer in self.clients.items():
            try:
                success = await _send_tasks_to_stream(tasks, writer)
                if success:
                    successful_sends += 1
                else:
                    failed_clients.append(client_id)
            except Exception as e:
                logging.error(f"Error sending to client {client_id}: {e}")
                failed_clients.append(client_id)

        # Clean up failed connections
        for client_id in failed_clients:
            if client_id in self.clients:
                writer = self.clients[client_id]
                writer.close()
                await writer.wait_closed()
                del self.clients[client_id]
                logging.info(f"Removed failed client {client_id}")

        return successful_sends

    def get_connected_clients(self) -> Set[str]:
        return set(self.clients.keys())

    async def stop(self):
        # Close all client connections
        for client_id, writer in self.clients.items():
            writer.close()
            await writer.wait_closed()
        self.clients.clear()

        # Stop server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.server = None

        logging.info("Server stopped")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


class NetworkClient:
    def __init__(self, host: str, port: int, timeout: float = 30.0):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(timeout)
        self.socket.connect((host, port))
        logging.info(f"Connected to server at {host}:{port}")

    def _send_tasks(self, tasks: dict[bytes, Task]) -> bool:
        if not self.socket:
            logging.error("Not connected to server")
            return False

        try:
            data = pickle.dumps(tasks)
            size = len(data)

            # Send size header (4 bytes)
            size_header = struct.pack('!I', size)
            self.socket.sendall(size_header)

            # Send data
            self.socket.sendall(data)

            return True
        except Exception as e:
            logging.error(f"Error sending tasks: {e}")
            return False

    def _recv_all(self, size: int) -> Optional[bytes]:
        buffer = b''
        while len(buffer) < size:
            try:
                chunk = self.socket.recv(size - len(buffer))
                if not chunk:
                    logging.error("Connection closed by peer")
                    return None
                buffer += chunk
            except socket.timeout:
                logging.error("Receive timeout")
                return None
            except Exception as e:
                logging.error(f"Error receiving data: {e}")
                return None
        return buffer

    def _receive_tasks(self) -> Optional[dict[bytes, Task]]:
        if not self.socket:
            logging.error("Not connected to server")
            return None

        try:
            # Receive size header (4 bytes)
            size_header = self._recv_all(4)
            if not size_header:
                return None

            size = struct.unpack('!I', size_header)[0]

            # Receive data
            data = self._recv_all(size)
            if not data:
                return None

            tasks = pickle.loads(data)
            return tasks

        except socket.timeout:
            logging.debug("Receive timeout")
            return None
        except Exception as e:
            logging.error(f"Error receiving tasks: {e}")
            return None

    def disconnect(self):
        if self.socket:
            self.socket.close()
            self.socket = None
            logging.info("Disconnected from server")

    def is_connected(self) -> bool:
        return self.socket is not None

    def sync(self, task_manager: SharedTaskManager) -> bool:
        if not self.is_connected():
            logging.error("Not connected to server")
            return False

        # Send local tasks to server
        if not self._send_tasks(task_manager._tasks):
            logging.error("Failed to send tasks to server")
            return False

        # Receive tasks from server
        received_tasks = self._receive_tasks()
        if received_tasks is None:
            logging.error("Failed to receive tasks from server")
            return False

        # Update local task manager with received tasks
        task_manager.update(received_tasks, self_is_priority=False)
        return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
