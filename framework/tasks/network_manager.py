import logging
import socket
import pickle
from typing import Optional

from ..prelude import *


class NetworkManager:
    def __init__(self):
        self.server_socket: Optional[socket.socket] = None
        self.is_listening: bool = False

    def start_listening(self, port: int, timeout: int = 30) -> bool:
        """Start listening on port for incoming UDP packets."""
        if self.is_listening:
            logging.warning("Already listening")
            return False

        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.settimeout(timeout)
            self.server_socket.bind(('0.0.0.0', port))
            self.is_listening = True
            logging.info(f"NetworkManager started listening on port {port} (timeout: {timeout}s)")
            return True
        except Exception as e:
            logging.error(f"Failed to start listening: {e}")
            return False

    def stop_listening(self):
        """Stop listening and close server socket."""
        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None
        self.is_listening = False
        logging.info("Stopped listening")

    def send_tasks(self, tasks: dict[bytes, Task], address: tuple[str, int]) -> None:
        """Send tasks to target address with chunking for large data."""
        data = pickle.dumps(tasks)
        total_size = len(data)

        # Send size header first
        size_header = total_size.to_bytes(8, byteorder='big')
        self.server_socket.sendto(size_header, address)
        self.server_socket.sendto(data, address)

    def receive_tasks(self) -> Optional[tuple[dict[bytes, Task], tuple[str, int]]]:
        """Receive tasks from UDP peer. Handles fragmented packets by chunking."""
        try:
            header_data, address = self.server_socket.recvfrom(8)  # 8 bytes for size header
            total_size = int.from_bytes(header_data, byteorder='big')

        except socket.timeout:
            logging.debug(f"Receive timeout")
            return None
        except socket.error as e:
            logging.error(f"Socket error while receiving header: {e}")
            return None

        buffer = b''
        try:
            while len(buffer) < total_size:
                chunk, _ = self.server_socket.recvfrom(min(1400, total_size - len(buffer)))
                buffer += chunk
        except socket.timeout:
            logging.error(f"Receive timeout")
            return None
        except Exception as e:
            logging.error(f"Error receiving tasks: {e}")
            return None

        try:
            tasks = pickle.loads(buffer)
        except Exception as e:
            logging.error(f"Error deserializing tasks: {e}")
            return None

        return tasks, address

    def __del__(self):
        """Cleanup socket on destruction."""
        self.stop_listening()
