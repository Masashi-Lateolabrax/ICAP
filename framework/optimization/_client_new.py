import dataclasses
import socket
import logging
import threading
import signal
import queue
import time
from typing import Optional, Callable

from ..prelude import *
from ..types.communication import Packet, PacketType
from ._connection_utils import send_packet, receive_packet, communicate


@dataclasses.dataclass
class CalculationState:
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
        logging.info(f"Connected to server: {server_address}:{port}")
    except (socket.error, socket.timeout) as e:
        logging.error(f"Connection error: {e}")
        raise ConnectionError(f"Failed to connect to server {server_address}:{port}") from e

    handshake_packet = Packet(_packet_type=PacketType.HANDSHAKE, data=None)
    if send_packet(sock, handshake_packet) != CommunicationResult.SUCCESS:
        logging.error("Failed to send handshake packet")
        sock.close()
        raise ConnectionError("Failed to send handshake packet")

    success, ack_packet = receive_packet(sock)
    if not success or ack_packet is None or ack_packet.packet_type != PacketType.ACK:
        logging.error("Failed to receive handshake ACK")
        sock.close()
        raise ConnectionError("Failed to receive handshake ACK")

    logging.info("Handshake completed successfully")
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
        logging.info("Evaluation worker started")

        while not self.stop_event.is_set():
            try:
                individuals = self.task_queue.get(timeout=1.0)
                if individuals is None:
                    logging.info("Received poison pill, stopping evaluation worker")
                    break
                self.response_queue.put(
                    CalculationState(idle=False, throughput=None)
                )

            except queue.Empty:
                self.response_queue.put(
                    CalculationState(idle=True)
                )
                continue

            logging.debug(f"Evaluation worker received {len(individuals)} individuals")

            for individual in individuals:
                if self.stop_event.is_set():
                    break

                try:
                    individual.timer_start()
                    fitness = self.evaluation_function(individual)
                    individual.timer_end()

                    individual.set_fitness(fitness)
                    individual.set_calculation_state(CalculationState.FINISHED)

                    self.response_queue.put(
                        CalculationState(
                            idle=False,
                            throughput=1 / (individual.get_elapse() + 1e-10)
                        )
                    )

                    logging.debug(f"Evaluated individual with fitness: {fitness}")

                except Exception as e:
                    logging.error(f"Error during evaluation function execution: {e}")
                    individual.set_fitness(float('inf'))
                    continue

            self.response_queue.put(
                CalculationState(
                    idle=True,
                    individuals=individuals,
                )
            )

        logging.info("Evaluation worker stopped")

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
            logging.info("Evaluation worker has been stopped")
        else:
            logging.warning("Evaluation worker is not running, cannot stop it")

    def add_task(self, individuals: list[Individual]) -> None:
        if not self.is_alive():
            raise RuntimeError("Evaluation worker is not running")
        self.task_queue.put(individuals)

    def get_response(self) -> CalculationState:
        if not self.is_alive():
            raise RuntimeError("Evaluation worker is not running")
        try:
            return self.response_queue.get(timeout=1.0)
        except queue.Empty:
            return CalculationState(idle=True)


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
        logging.debug("Sending request packet to server")

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

        logging.debug(f"Sent {len(self.evaluated_task)} individuals to server")
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
        logging.info(f"Received signal {signum}, stopping client...")
        stop_event.set()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    sock = _connect_to_server(server_address, port)

    evaluation_worker = _EvaluationWorker(evaluation_function)
    evaluation_worker.run()

    communication_worker = _CommunicationWorker(sock)

    while not stop_event.is_set():
        # Update throughput from evaluation worker
        calc_state = evaluation_worker.get_response()
        if calc_state.throughput is not None:
            communication_worker.set_throughput(calc_state.throughput)

        if calc_state.individuals is not None:
            communication_worker.set_evaluated_task(calc_state.individuals)

        # Run communication worker and get new task
        new_task = communication_worker.run()
        if new_task is not None:
            evaluation_worker.add_task(new_task)

        time.sleep(0.01)

    logging.info("Client disconnected")
