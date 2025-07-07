import socket
import pickle
import struct
import logging
from typing import Optional

from ..prelude import *


def _send_message(sock: socket.socket, data: bytes, attempt_count: int = 10) -> CommunicationResult:
    size = struct.pack('!I', len(data))

    for attempt in range(attempt_count):
        try:
            sock.sendall(size + data)
            return CommunicationResult.SUCCESS

        except socket.timeout:
            logging.warning(f"Socket timeout on attempt {attempt + 1} while sending message")
            continue

        except (socket.error, struct.error, BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
            logging.error(f"Error sending message: {e}")
            return CommunicationResult.CONNECTION_ERROR

    logging.error("Failed to send message after multiple attempts")
    return CommunicationResult.OVER_ATTEMPT_COUNT


def _receive_bytes(sock: socket.socket, size: int, blocking=True) -> tuple[_CommunicationResult, Optional[bytes]]:
    attempt = 0
    data = b''
    while len(data) < size:
        try:
            try:
                sock.getpeername()
            except (socket.error, OSError):
                logging.info("Socket disconnected (peer unreachable)")
                return _CommunicationResult.DISCONNECTED, None
            
            chunk = sock.recv(size - len(data))
            if not chunk:
                logging.info("Server disconnected")
                return _CommunicationResult.DISCONNECTED, None
            data += chunk

        except socket.timeout:
            if not blocking:
                logging.info("Socket timeout while receiving data, returning None")
                return _CommunicationResult.OVER_ATTEMPT_COUNT, None

            attempt += 1
            logging.warning(f"Socket timeout on attempt {attempt} while receiving data")
            if attempt >= ATTEMPT_COUNT:
                logging.error("Exceeded maximum attempt count while receiving data")
                return _CommunicationResult.OVER_ATTEMPT_COUNT, None

            continue

        except (socket.error, struct.error, ConnectionResetError, ConnectionAbortedError) as e:
            logging.error(f"Connection error while receiving data: {e}")
            return _CommunicationResult.CONNECTION_ERROR, None

    return _CommunicationResult.SUCCESS, data


def send_individuals(sock: socket.socket, individuals: list[Individual]) -> bool:
    """Send a list of Individual objects over socket."""
    try:
        data = pickle.dumps(individuals)
        result = _send_message(sock, data)
        if result != _CommunicationResult.SUCCESS:
            logging.error("Connection error while sending individual list")
        return result == _CommunicationResult.SUCCESS

    except pickle.PicklingError as e:
        logging.error(f"Pickle error while sending individual list: {e}")
        return False


def receive_individuals(sock: socket.socket, blocking=True) -> tuple[bool, Optional[list[Individual]]]:
    """Receive a list of Individual objects from socket with proper error handling.
    
    Returns:
        tuple[bool, Optional[list[Individual]]]: Status and received objects:
            - (True, list[Individual]): Successfully received and deserialized Individual objects
            - (True, None): Operation timed out (exceeded retry attempts) - recoverable
            - (False, None): Fatal error occurred (connection, protocol, or deserialization error)
    """
    # First, receive the size
    result, raw = _receive_bytes(sock, 4, blocking=blocking)
    if result == _CommunicationResult.OVER_ATTEMPT_COUNT:
        return True, None
    elif result != _CommunicationResult.SUCCESS:
        return False, None

    # Extract size from the 4-byte data
    try:
        size_data = struct.unpack('!I', raw)[0]
    except struct.error as e:
        logging.error(f"Error unpacking size data: {e}")
        return False, None

    # Receive the actual data
    result, raw = _receive_bytes(sock, size_data, blocking=blocking)
    if result == _CommunicationResult.OVER_ATTEMPT_COUNT:
        return True, None
    elif result != _CommunicationResult.SUCCESS:
        return False, None

    # Deserialize the Individual list
    try:
        individuals = pickle.loads(raw)
        return True, individuals
    except pickle.UnpicklingError as e:
        logging.error(f"Pickle error while receiving individual list: {e}")
        return False, None
