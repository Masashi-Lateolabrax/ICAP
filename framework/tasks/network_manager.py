import logging
import socket
import pickle
from typing import Optional

from ..prelude import *


class NetworkManager:
    def __init__(self):
        self.socket: Optional[socket.socket] = None

    def start_communication(self, port: int, timeout: int = 30) -> bool:
        if self.socket:
            logging.warning("Already binding")
            return False

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.settimeout(timeout)
            self.socket.bind(('0.0.0.0', port))
            logging.info(f"NetworkManager started on port {port} (timeout: {timeout}s)")
            return True
        except Exception as e:
            logging.error(f"Failed to start: {e}")
            return False

    def stop_communication(self):
        if self.socket:
            self.socket.close()
            self.socket = None
        logging.info("Stopped")

    def send_tasks(self, tasks: dict[bytes, Task], address: tuple[str, int]) -> None:
        sock = self.socket if self.socket else socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        data = pickle.dumps(tasks)
        total_size = len(data)

        # Send size header first
        size_header = total_size.to_bytes(8, byteorder='big')
        sock.sendto(size_header, address)
        sock.sendto(data, address)

    def receive_tasks(self) -> Optional[tuple[dict[bytes, Task], tuple[str, int]]]:
        try:
            header_data, address = self.socket.recvfrom(8)  # 8 bytes for size header
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
                chunk, _ = self.socket.recvfrom(min(1400, total_size - len(buffer)))
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
        self.stop_communication()
