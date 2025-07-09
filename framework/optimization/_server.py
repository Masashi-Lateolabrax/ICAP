import socket
import threading
import logging
import time
import select
from queue import Queue
from typing import Optional, Callable

from icecream import ic

from ._cmaes import CMAES
from ._distribution import Distribution
from ..prelude import *
from ..types.communication import Packet, PacketType
from ._connection_utils import send_packet, receive_packet

# Socket timeout for dead connection detection in seconds
SOCKET_TIMEOUT = 45


class _Server:
    def __init__(
            self,
            settings: Settings,
            socket_queue: Queue,
            stop_event: threading.Event,
            handler: Optional[Callable[[CMAES, list[Individual]], None]] = None
    ):
        self.settings = settings

        self.socket_queue = socket_queue
        self.stop_event = stop_event
        self.handler = handler

        self.sockets: list[socket.socket] = []
        self.socket_states: dict[socket.socket, SocketState] = {}

    def sock_name(self, sock: socket) -> str:
        if sock in self.socket_states:
            address = self.socket_states[sock].address
        else:
            try:
                address = f"{sock.getpeername()[0]}:{sock.getpeername()[1]}"
            except socket.error as e:
                logging.error(f"Error getting peer name: {e}")
                address = "Unknown"
        return address

    def _drop_socket(self, sock: socket.socket):
        if sock in self.sockets:
            self.sockets.remove(sock)

            if sock in self.socket_states:
                status = self.socket_states[sock]
                if status.assigned_individuals:
                    count = 0
                    for i in status.assigned_individuals:
                        if not i.is_finished:
                            i.set_calculation_state(CalculationState.CORRUPTED)
                            count += 1
                    logging.warning(f"Dropped {count} unfinished individuals from socket: {status.address}")
                del self.socket_states[sock]

            sock.close()
        else:
            logging.warning(f"Attempted to drop a socket that is not in the list: {self.sock_name(sock)}")

    def _mut_drop_dead_sockets(self):
        current_time = time.time()
        for sock, state in list(self.socket_states.items()):
            if current_time - state.last_heartbeat > SOCKET_TIMEOUT:
                logging.warning(f"Dropping dead socket: {state.address} (no heartbeat for {SOCKET_TIMEOUT} seconds)")
                self._drop_socket(sock)

    def _mut_retrieve_socket(self) -> Optional[socket.socket]:
        while not self.socket_queue.empty():
            try:
                sock = self.socket_queue.get_nowait()
                if isinstance(sock, socket.socket):
                    self.sockets.append(sock)
                    self.socket_states[sock] = SocketState(sock)
            except Exception as e:
                logging.error(f"Error retrieving socket from queue: {e}")

    def _receive_packet(self, timeout=1) -> Optional[dict[PacketType, dict[socket.socket, Packet]]]:
        try:
            readable, _, _ = select.select(self.sockets, [], self.sockets, timeout)

        except select.error as e:
            logging.error(f"Select error: {e}")
            return None

        if not readable:
            return None

        result = {}
        for sock in readable:
            packet = None
            while True:
                success, packet_ = receive_packet(sock)
                ic(success)

                if success == CommunicationResult.TIMEOUT or success == CommunicationResult.OVER_ATTEMPT_COUNT:
                    logging.warning(f"Received packet from socket: {self.sock_name(sock)}")
                    break
                elif success != CommunicationResult.SUCCESS:
                    logging.error(f"Failed to receive packet from {self.sock_name(sock)}: {success}")
                    self._drop_socket(sock)
                    break

                ic(packet_.packet_type)
                packet = packet_

            if packet is None:
                logging.error(f"Invalid packet received from {self.sock_name(sock)}")
                continue
            if sock not in self.socket_states:
                logging.error(f"This socket({self.sock_name(sock)}) may be dropped, so it is not in socket states")
                continue

            self.socket_states[sock].last_heartbeat = ic(time.time())
            if packet.packet_type not in result:
                result[packet.packet_type] = {}
            result[packet.packet_type][sock] = packet

        return result

    def _response_ack(self, sock: socket.socket, data=None):
        response_packet = Packet(PacketType.ACK, data=data)
        if send_packet(sock, response_packet, retry=3) != CommunicationResult.SUCCESS:
            logging.error(f"Failed to send ACK packet to {self.sock_name(sock)}")
            self._drop_socket(sock)

    def _deal_with_handshake(self, handshake_packets: dict[socket.socket, Packet]):
        for sock, packet in handshake_packets.items():
            if sock not in self.socket_states:
                logging.warning(f"Socket {self.sock_name(sock)} not found in socket states")
                continue
            self._response_ack(sock)

    def _deal_with_heartbeat(self, heartbeat_packets: dict[socket.socket, Packet]):
        for sock, packet in heartbeat_packets.items():
            if sock not in self.socket_states:
                logging.warning(f"Socket {self.sock_name(sock)} not found in socket states")
                continue
            self.socket_states[sock].throughput = ic(packet.data)
            self._response_ack(sock)

    def _deal_with_response(self, response_packets: dict[socket.socket, Packet]):
        for sock, packet in response_packets.items():
            if sock not in self.socket_states:
                logging.warning(f"Socket {self.sock_name(sock)} not found in socket states")
                continue
            if not isinstance(packet.data, list):
                logging.error(f"Invalid data type in RESPONSE packet from {self.sock_name(sock)}: {type(packet.data)}")
                self._drop_socket(sock)
                continue
            if len(packet.data) != len(self.socket_states[sock].assigned_individuals):
                logging.error(f"Size mismatch in RESPONSE packet from {self.sock_name(sock)}")
                self._drop_socket(sock)
                continue
            for i, evaluated_individual in enumerate(packet.data):
                if not isinstance(evaluated_individual, Individual):
                    logging.error(f"Invalid individual in RESPONSE packet from {self.sock_name(sock)}")
                    self._drop_socket(sock)
                    continue
                self.socket_states[sock].assigned_individuals[i].copy_from(evaluated_individual)
            self.socket_states[sock].assigned_individuals = None
            self._response_ack(sock)

    def _deal_with_request(self, request_packets: dict[socket.socket, Packet]):
        for sock, packet in request_packets.items():
            if sock not in self.socket_states:
                logging.warning(f"Socket {self.sock_name(sock)} not found in socket states")
                continue
            self._response_ack(sock, data=self.socket_states[sock].assigned_individuals)

    def _update_assigned_individuals(self, cmaes: CMAES, distribution: Distribution):
        for sock in self.sockets:
            if self.socket_states[sock].assigned_individuals is not None:
                continue
            batch_size = ic(distribution.get_batch_size(sock))
            self.socket_states[sock].assigned_individuals = cmaes.get_individuals(batch_size)

    def run(self):
        distribution = Distribution()
        cmaes = CMAES(
            max_generation=self.settings.Optimization.GENERATION,
            dimension=self.settings.Optimization.DIMENSION,
            sigma=self.settings.Optimization.SIGMA,
            population_size=self.settings.Optimization.POPULATION,
        )

        while not self.stop_event.is_set():
            self._mut_retrieve_socket()
            if ic(len(self.sockets)) == 0:
                time.sleep(1)
                continue

            sorted_packets: Optional[dict[PacketType, dict[socket.socket, Packet]]] = self._receive_packet()
            if sorted_packets is None:
                continue

            self._deal_with_handshake(sorted_packets.get(PacketType.HANDSHAKE, {}))
            self._deal_with_heartbeat(sorted_packets.get(PacketType.HEARTBEAT, {}))
            self._deal_with_response(sorted_packets.get(PacketType.RESPONSE, {}))
            self._deal_with_request(sorted_packets.get(PacketType.REQUEST, {}))

            for sock in sorted_packets.get(PacketType.DISCONNECTION, {}).keys():
                self._drop_socket(sock)
            self._mut_drop_dead_sockets()

            result, individuals = cmaes.update()
            ic(result, len(individuals))

            if self.handler:
                self.handler(cmaes, individuals)

            distribution.update(len(individuals), self.socket_states)

            self._update_assigned_individuals(cmaes, distribution)


def _spawn_thread(
        settings: Settings, handler: Optional[Callable[[CMAES, list[Individual]], None]] = None
) -> tuple[threading.Thread, Queue, threading.Event]:
    socket_queue: Queue = Queue()
    stop_event: threading.Event = threading.Event()
    thread = threading.Thread(
        target=lambda: _Server(settings, socket_queue, stop_event, handler).run()
    )
    thread.daemon = True
    thread.start()
    return thread, socket_queue, stop_event


def _create_server_socket(host: str, port: int) -> socket.socket:
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.settimeout(1.0)
        server_socket.bind((host, port))
        ic(host, port)
        return server_socket
    except socket.error as e:
        ic(e)
        raise RuntimeError(f"Failed to create server socket: {e}")


def _server_entrance(host: str, port: int, socket_queue: Queue, stop_event: threading.Event):
    sock = _create_server_socket(host, port)
    sock.listen(5)

    try:
        while not stop_event.is_set():
            try:
                client_socket, client_address = sock.accept()
                client_socket.settimeout(0.1)
                ic(client_address)
                socket_queue.put(client_socket)

            except socket.timeout:
                continue

            except socket.error as e:
                ic(e)
                continue

    finally:
        try:
            sock.close()
        except Exception as e:
            ic(e)


class OptimizationServer:
    def __init__(self, settings: Settings, handler: Optional[Callable[[CMAES, list[Individual]], None]] = None):
        self.settings = settings
        self.handler = handler

    def start_server(self) -> None:
        server_thread, queue, stop_event = _spawn_thread(self.settings, self.handler)
        _server_entrance(self.settings.Server.HOST, self.settings.Server.PORT, queue, stop_event)
        server_thread.join()
