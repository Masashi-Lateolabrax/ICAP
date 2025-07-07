import dataclasses
import socket
import logging
import threading
import signal
import queue
import time
from typing import Optional, Callable

from icecream import ic
from ..prelude import *
from ..types.communication import Packet, PacketType
from ._connection_utils import send_packet, communicate


@dataclasses.dataclass
class ClientCalculationState:
    idle: bool
    throughput: Optional[float] = None
    individuals: Optional[list[Individual]] = None
    error: Optional[str] = None

    @property
    def is_idle(self) -> bool:
        return self.idle


def _connect_to_server(server_address: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10.0)

    try:
        sock.connect((server_address, port))

    except (socket.error, socket.timeout) as e:
        raise ConnectionError(f"Failed to connect to server {server_address}:{port}") from e

    handshake_packet = Packet(_packet_type=PacketType.HANDSHAKE, data=None)
    result, ack_packet = communicate(sock, handshake_packet)
    if result != CommunicationResult.SUCCESS:
        sock.close()
        raise ConnectionError("Failed to complete handshake")

    return sock


class _EvaluationWorker:
    def __init__(
            self,
            evaluation_function: EvaluationFunction,
    ):
        self.task_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.evaluation_function = evaluation_function
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def _worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                individuals = self.task_queue.get(timeout=1.0)
                if individuals is None:
                    break
                self.response_queue.put(
                    ClientCalculationState(idle=False, throughput=None)
                )

            except queue.Empty:
                self.response_queue.put(
                    ClientCalculationState(idle=True)
                )
                continue

            for individual in individuals:
                if self.stop_event.is_set():
                    break

                try:
                    individual.timer_start()
                    fitness = self.evaluation_function(individual)
                    individual.timer_end()

                    individual.set_fitness(fitness)
                    individual.set_calculation_state(CalculationState.FINISHED)

                    throughput = 1 / (individual.get_elapse() + 1e-10)
                    self.response_queue.put(
                        ClientCalculationState(
                            idle=False,
                            throughput=throughput
                        )
                    )

                except Exception as e:
                    logging.error(f"Error during evaluation function execution: {e}")
                    individual.set_fitness(float('inf'))
                    continue

            self.response_queue.put(
                ClientCalculationState(
                    idle=True,
                    individuals=individuals,
                )
            )

    def is_alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def run(self):
        self.thread = threading.Thread(target=self._worker)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        if self.is_alive():
            self.stop_event.set()
            self.thread.join(timeout=5.0)
        else:
            logging.warning("Evaluation worker is not running, cannot stop it")

    def add_task(self, individuals: list[Individual]) -> None:
        if not self.is_alive():
            raise RuntimeError("Evaluation worker is not running")
        self.task_queue.put(individuals)

    def get_response(self) -> ClientCalculationState:
        if not self.is_alive():
            raise RuntimeError("Evaluation worker is not running")
        try:
            return self.response_queue.get(timeout=1.0)
        except queue.Empty:
            return ClientCalculationState(idle=True)


class _CommunicationWorker:
    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.last_heartbeat = time.time()
        self.heartbeat_interval = 10  # seconds
        self.throughput = 0.0
        self.task: Optional[list[Individual]] = None
        self.evaluated_task: Optional[list[Individual]] = None

    def is_alive(self) -> bool:
        return self.sock is not None

    def is_assigned(self) -> bool:
        return self.task is not None or self.evaluated_task is not None

    def set_evaluated_task(self, individuals: list[Individual]) -> None:
        self.task = None
        self.evaluated_task = individuals

    def _heartbeat(self, throughput: float) -> CommunicationResult:
        if not self.is_alive():
            logging.error("Socket is not alive, cannot send heartbeat")
            return CommunicationResult.CONNECTION_ERROR

        time_since_last = time.time() - self.last_heartbeat
        if time.time() - self.last_heartbeat < self.heartbeat_interval:
            return CommunicationResult.SUCCESS

        packet = Packet(_packet_type=PacketType.HEARTBEAT, data=throughput)
        result, packet = communicate(self.sock, packet)

        if result != CommunicationResult.SUCCESS:
            logging.error("Failed to send heartbeat packet")
            return result

        self.last_heartbeat = time.time()

        return CommunicationResult.SUCCESS

    def _request(self) -> CommunicationResult:
        if not self.is_alive():
            logging.error("Socket is not alive, cannot send request")
            return CommunicationResult.CONNECTION_ERROR

        if self.is_assigned():
            logging.error("Already assigned, cannot send request")
            return CommunicationResult.SUCCESS

        packet = Packet(_packet_type=PacketType.REQUEST, data=None)
        result, packet = communicate(self.sock, packet)

        if result != CommunicationResult.SUCCESS:
            logging.error("Failed to send request packet")
            return result

        self.task = packet.data

        return CommunicationResult.SUCCESS

    def _return(self) -> CommunicationResult:
        if not self.is_alive():
            logging.error("Socket is not alive, cannot send individuals")
            return CommunicationResult.CONNECTION_ERROR

        if not self.is_assigned():
            logging.error("No individuals to return, cannot send response")
            return CommunicationResult.SUCCESS

        packet = Packet(_packet_type=PacketType.RESPONSE, data=self.evaluated_task)
        result, packet = communicate(self.sock, packet)

        if result != CommunicationResult.SUCCESS:
            logging.error("Failed to send response packet with individuals")
            return result

        self.evaluated_task = None

        return CommunicationResult.SUCCESS

    def set_throughput(self, throughput: float) -> None:
        self.throughput = throughput

    def run(self) -> Optional[list[Individual]]:
        if not self.is_alive():
            raise RuntimeError("Communication worker is not running")

        self._heartbeat(self.throughput)
        if not self.is_assigned():
            self._request()

        if self.task is None and self.evaluated_task is not None:
            self._return()

        return self.task


def connect_to_server(
        server_address: str,
        port: int,
        evaluation_function: EvaluationFunction,
        handler: Optional[Callable] = None
) -> None:
    stop_event = threading.Event()

    def signal_handler(signum, frame):
        logging.warning(f"Received signal {signum}, stopping client...")
        stop_event.set()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    sock = _connect_to_server(server_address, port)

    evaluation_worker = _EvaluationWorker(evaluation_function)
    evaluation_worker.run()

    communication_worker = _CommunicationWorker(sock)

    try:
        iteration = 0
        while not stop_event.is_set():
            calc_state = evaluation_worker.get_response()
            if calc_state.throughput is not None:
                communication_worker.set_throughput(calc_state.throughput)

            if calc_state.individuals is not None:
                communication_worker.set_evaluated_task(calc_state.individuals)

            try:
                new_task = communication_worker.run()
                evaluation_worker.add_task(new_task)

            except Exception as e:
                logging.error(f"Communication error: {e}")
                break

            time.sleep(0.01)

    finally:
        evaluation_worker.stop()
        if sock:
            try:
                disconnect_packet = Packet(_packet_type=PacketType.DISCONNECTION, data=None)
                send_packet(sock, disconnect_packet)
                sock.close()
            except Exception as e:
                logging.warning(f"Error during disconnection: {e}")
