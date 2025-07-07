import socket
import threading
import logging
import time
import select
from datetime import datetime
from queue import Queue
import math
from typing import Optional, Callable

import numpy as np
from icecream import ic

from ._cmaes import CMAES
from ._distribution import Distribution
from ..prelude import *
from ..types.communication import Packet, PacketType
from ._connection_utils import send_packet, receive_packet


class Reporter:
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.buffer: list[str] = []
        self.last_output_time = time.time()

    def add(self, address, fitness_values: list[float], throughput: float) -> None:
        if not fitness_values:
            logging.warning(f"No fitness values provided for address {address}")
            return

        timestamp = datetime.now().strftime("%H:%M:%S")

        average_fitness = sum(fitness_values) / len(fitness_values)
        variance_fitness = math.sqrt(sum((f - average_fitness) ** 2 for f in fitness_values) / len(fitness_values))
        error_count = sum(1 for f in fitness_values if f == float("inf"))

        message = (f"[{timestamp}] [{address}] "
                   f"Num:{len(fitness_values)}, "
                   f"AveFitness:{average_fitness:.2f}, "
                   f"Variance:{variance_fitness:.2f}, "
                   f"Error:{error_count}, "
                   f"Throughput:{throughput:.1f} ind/sec")
        self.buffer.append(message)

    def should_output(self) -> bool:
        current_time = time.time()
        return current_time - self.last_output_time >= self.interval

    def output(self) -> None:
        if self.buffer:
            for message in self.buffer:
                print(message)
            self.buffer.clear()
            self.last_output_time = time.time()


class _Server:
    def __init__(
            self,
            settings: Settings,
            socket_queue: Queue,
            stop_event: threading.Event,
            handler: Optional[Callable[[CMAES], None]] = None
    ):
        self.settings = settings

        self.socket_queue = socket_queue
        self.stop_event = stop_event
        self.handler = handler

        self.sockets: list[socket.socket] = []
        self.socket_states: dict[socket.socket, SocketState] = {}

        self.reporter = Reporter()

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
            logging.warning(f"Attempted to drop a socket that is not in the list: {sock.getpeername()}")

    def _mut_drop_dead_sockets(self):
        current_time = time.time()
        for sock, state in list(self.socket_states.items()):
            if current_time - state.last_heartbeat > 10:
                logging.warning(f"Dropping dead socket: {state.address} (no heartbeat for 10 seconds)")
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

    def _communicate_with_client(self, timeout: float = 20.0):
        try:
            readable, _, _ = select.select(self.sockets, [], self.sockets, timeout)

        except select.error as e:
            logging.error(f"Select error: {e}")
            return

        if not readable:
            return

        for sock in readable:
            success, packet = receive_packet(sock)
            ic(success, packet.packet_type)
            if success != CommunicationResult.SUCCESS:
                self._drop_socket(sock)
                continue

            self.socket_states[sock].last_heartbeat = time.time()

            response_packet = Packet(PacketType.ACK)
            match packet.packet_type:
                case PacketType.HEARTBEAT:
                    self.socket_states[sock].throughput = ic(packet.data)

                case PacketType.HANDSHAKE:
                    pass

                case PacketType.REQUEST:
                    response_packet.data = self.socket_states[sock].assigned_individuals
                    ic(len(response_packet.data) if response_packet.data else None)

                case PacketType.RESPONSE:
                    if not isinstance(packet.data, list):
                        logging.error(f"Invalid data type in RESPONSE packet: {type(packet.data)}")
                        self._drop_socket(sock)
                        continue
                    if not packet.data:
                        logging.warning(f"Received empty RESPONSE packet from {sock.getpeername()}")
                        continue

                    fitness_values = []
                    throughput = 0.0
                    for i in packet.data:
                        if not isinstance(i, Individual):
                            logging.error(f"Invalid individual in RESPONSE packet: {i}")
                            break
                        for j in self.socket_states[sock].assigned_individuals:
                            if j.is_finished:
                                continue
                            if np.array_equal(i, j):
                                j.copy_from(i)
                                continue
                        fitness_values.append(i.get_fitness())
                        throughput += 1 / (i.get_elapse() + 1e-10)

                    ic(sock, np.average(fitness_values), throughput)
                    self.reporter.add(sock.fileno(), fitness_values, throughput / len(packet.data))
                    self.socket_states[sock].assigned_individuals = None

                case PacketType.DISCONNECTION:
                    send_packet(sock, response_packet)
                    self._drop_socket(sock)
                    continue

                case PacketType.ACK:
                    logging.warning(f"Received ACK packet: {packet.data}")
                    continue

            result = ic(send_packet(sock, response_packet))
            if result != CommunicationResult.SUCCESS:
                self._drop_socket(sock)

    def _update_assigned_individuals(self, cmaes: CMAES, distribution: Distribution):
        for sock in self.sockets:
            if self.socket_states[sock].assigned_individuals is not None:
                continue
            batch_size = ic(distribution.get_batch_size(sock))
            self.socket_states[sock].assigned_individuals = cmaes.get_individuals(batch_size)

        if self.reporter.should_output():
            self.reporter.output()

    def run(self):
        distribution = Distribution()
        cmaes = CMAES(
            dimension=self.settings.Optimization.dimension,
            sigma=self.settings.Optimization.sigma,
            population_size=self.settings.Optimization.population_size,
        )

        while not self.stop_event.is_set():
            self._mut_retrieve_socket()
            if ic(len(self.sockets)) == 0:
                time.sleep(1)
                continue

            self._mut_drop_dead_sockets()
            self._communicate_with_client()

            result, individuals = cmaes.update()
            ic(result, len(individuals))

            distribution.update(len(individuals), self.socket_states)

            self._update_assigned_individuals(cmaes, distribution)

        self.reporter.output()


def _spawn_thread(
        settings: Settings, handler: Optional[Callable[[CMAES], None]] = None
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
    def __init__(self, settings: Settings):
        self.settings = settings

    def start_server(self, handler: Optional[Callable[[CMAES], None]] = None) -> None:
        server_thread, queue, stop_event = _spawn_thread(self.settings, handler)
        _server_entrance(self.settings.Server.HOST, self.settings.Server.PORT, queue, stop_event)
        server_thread.join()
