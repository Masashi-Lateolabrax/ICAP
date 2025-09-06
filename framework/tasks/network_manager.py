import logging
import asyncio
import pickle
import struct
import socket
from typing import Optional
from enum import Enum

from icecream import ic
from ..prelude import *
from .shared_task_manager import SharedTaskManager


class ReceiveStatus(Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class NetworkServer:
    def __init__(self, host: str, port: int, timeout: float = 30.0):
        ic(host, port, timeout)
        self.host = host
        self.port = port
        self.timeout = timeout
        self.task_manager = SharedTaskManager()
        self.server = None

    async def _create_server(self):
        if self.server is None:
            self.server = await asyncio.start_server(self._handle_client, self.host, self.port)
            ic(self.server)
            logging.info(f"TCP server created on {self.host}:{self.port}")
        return self.server

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        client_address = writer.get_extra_info('peername')
        client_id = f"{client_address[0]}:{client_address[1]}"
        ic(client_id)
        logging.info(f"Client {client_id} connected")

        try:
            while True:
                tasks, status = await self._receive_tasks(reader)
                ic(len(tasks) if tasks else 0, status)

                if status == ReceiveStatus.DISCONNECTED or status == ReceiveStatus.ERROR:
                    # Client disconnected or error occurred - break the loop
                    break
                elif status == ReceiveStatus.TIMEOUT:
                    # Healthy timeout - continue waiting
                    await asyncio.sleep(0.1)
                    continue
                elif status == ReceiveStatus.SUCCESS and tasks:
                    # Successfully received tasks - process them
                    ic(len(self.task_manager._tasks))
                    self.task_manager.update(tasks, self_is_priority=True)
                    ic(len(self.task_manager._tasks))
                    # Send server's tasks back to client
                    await self._send_tasks(self.task_manager._tasks, writer)

        except Exception as e:
            logging.error(f"Error handling client {client_id}: {e}")
        finally:
            logging.info(f"Client {client_id} disconnected")
            writer.close()
            await writer.wait_closed()

    async def _send_tasks(self, tasks: dict[bytes, Task], writer: asyncio.StreamWriter) -> bool:
        ic(len(tasks))
        if not writer:
            logging.error("Writer not available")
            return False

        try:
            data = pickle.dumps(tasks)
            size = len(data)
            ic(size)

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

    async def _recv_all(self, reader: asyncio.StreamReader, size: int) -> Optional[bytes]:
        buffer = b''
        while len(buffer) < size:
            try:
                chunk = await asyncio.wait_for(
                    reader.read(size - len(buffer)),
                    timeout=self.timeout
                )
                if not chunk:
                    logging.error("Connection closed by peer")
                    return None
                buffer += chunk
            except asyncio.TimeoutError:
                logging.error("Receive timeout")
                return None
            except Exception as e:
                logging.error(f"Error receiving data: {e}")
                return None
        return buffer

    async def _receive_tasks(self, reader: asyncio.StreamReader) -> tuple[Optional[dict[bytes, Task]], ReceiveStatus]:
        if not reader:
            logging.error("Reader not available")
            return None, ReceiveStatus.ERROR

        try:
            # Receive size header (4 bytes)
            size_header = await self._recv_all(reader, 4)
            if not size_header:
                return None, ReceiveStatus.DISCONNECTED

            size = struct.unpack('!I', size_header)[0]
            ic(size)

            # Receive data
            data = await self._recv_all(reader, size)
            if not data:
                return None, ReceiveStatus.DISCONNECTED

            tasks = pickle.loads(data)
            ic(len(tasks))
            return tasks, ReceiveStatus.SUCCESS

        except asyncio.TimeoutError:
            logging.debug("Receive timeout - client still connected")
            return None, ReceiveStatus.TIMEOUT
        except asyncio.IncompleteReadError:
            logging.info("Client disconnected")
            return None, ReceiveStatus.DISCONNECTED
        except Exception as e:
            logging.error(f"Error receiving tasks: {e}")
            return None, ReceiveStatus.ERROR

    def is_connected(self) -> bool:
        return self.server is not None

    async def stop(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.server = None
        logging.info("Server stopped")

    async def __aenter__(self):
        await self._create_server()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


class NetworkClient:
    def __init__(self, host: str, port: int, timeout: float = 30.0):
        ic(host, port, timeout)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(timeout)
        self.socket.connect((host, port))
        ic(self.socket)
        logging.info(f"Connected to server at {host}:{port}")

    def _send_tasks(self, tasks: dict[bytes, Task]) -> bool:
        ic(len(tasks))
        if not self.socket:
            logging.error("Not connected to server")
            return False

        try:
            data = pickle.dumps(tasks)
            size = len(data)
            ic(size)

            # Send size header (4 bytes)
            size_header = struct.pack('!I', size)
            self.socket.sendall(size_header)

            # Send data
            self.socket.sendall(data)

            return True
        except Exception as e:
            logging.error(f"Error sending tasks: {e}")
            return False

    def _recv_all(self, size: int) -> tuple[Optional[bytes], ReceiveStatus]:
        buffer = b''
        while len(buffer) < size:
            try:
                chunk = self.socket.recv(size - len(buffer))
                if not chunk:
                    logging.error("Connection closed by peer")
                    return None, ReceiveStatus.DISCONNECTED
                buffer += chunk
            except socket.timeout:
                logging.error("Receive timeout")
                return None, ReceiveStatus.TIMEOUT
            except Exception as e:
                logging.error(f"Error receiving data: {e}")
                return None, ReceiveStatus.ERROR
        return buffer, ReceiveStatus.SUCCESS

    def _receive_tasks(self) -> tuple[Optional[dict[bytes, Task]], ReceiveStatus]:
        if not self.socket:
            logging.error("Not connected to server")
            return None, ReceiveStatus.ERROR

        # Receive size header (4 bytes)
        size_header, status = self._recv_all(4)
        if status != ReceiveStatus.SUCCESS:
            return None, status

        size = struct.unpack('!I', size_header)[0]
        ic(size)

        # Receive data
        data, status = self._recv_all(size)
        if status != ReceiveStatus.SUCCESS:
            return None, status

        tasks = pickle.loads(data)
        if not isinstance(tasks, dict):
            logging.error("Received data is not a valid task dictionary")
            return None, ReceiveStatus.ERROR

        ic(len(tasks))
        return tasks, ReceiveStatus.SUCCESS

    def disconnect(self):
        if self.socket:
            self.socket.close()
            self.socket = None
            logging.info("Disconnected from server")

    def is_connected(self) -> bool:
        return self.socket is not None

    def sync(self, task_manager: SharedTaskManager) -> bool:
        ic(self.is_connected())
        if not self.is_connected():
            logging.error("Not connected to server")
            return False

        # Send local tasks to server
        ic(len(task_manager._tasks))
        if not self._send_tasks(task_manager._tasks):
            logging.error("Failed to send tasks to server")
            return False

        # Receive tasks from server
        received_tasks, status = self._receive_tasks()
        ic(len(received_tasks) if received_tasks else 0, status)
        if status != ReceiveStatus.SUCCESS:
            logging.error(f"Failed to receive tasks from server: {status.value}")
            return False

        # Update local task manager with received tasks
        task_manager.update(received_tasks, self_is_priority=False)
        ic(len(task_manager._tasks))
        return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
