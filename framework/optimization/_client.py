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
    ic(f"_connect_to_server: Connecting to {server_address}:{port}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10.0)
    ic("_connect_to_server: Socket created with 10s timeout")

    try:
        sock.connect((server_address, port))
        ic("_connect_to_server: TCP connection established")
        logging.info(f"Connected to server: {server_address}:{port}")
    except (socket.error, socket.timeout) as e:
        ic("_connect_to_server: Connection failed:", e)
        logging.error(f"Connection error: {e}")
        raise ConnectionError(f"Failed to connect to server {server_address}:{port}") from e

    ic("_connect_to_server: Sending handshake packet")
    handshake_packet = Packet(_packet_type=PacketType.HANDSHAKE, data=None)
    result, ack_packet = communicate(sock, handshake_packet)
    ic("_connect_to_server: Handshake result:", result, "ack:", ack_packet)
    if result != CommunicationResult.SUCCESS:
        ic("_connect_to_server: Handshake failed")
        logging.error("Failed to complete handshake")
        sock.close()
        raise ConnectionError("Failed to complete handshake")

    ic("_connect_to_server: Handshake completed successfully")
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
        ic("_worker: Evaluation worker started")
        logging.info("Evaluation worker started")

        while not self.stop_event.is_set():
            try:
                ic("_worker: Waiting for task from queue...")
                individuals = self.task_queue.get(timeout=1.0)
                ic("_worker: Received task:", individuals)
                if individuals is None:
                    ic("_worker: Received poison pill, stopping")
                    logging.info("Received poison pill, stopping evaluation worker")
                    break
                ic("_worker: Setting state to not idle")
                self.response_queue.put(
                    ClientCalculationState(idle=False, throughput=None)
                )

            except queue.Empty:
                ic("_worker: Task queue empty, setting idle state")
                self.response_queue.put(
                    ClientCalculationState(idle=True)
                )
                continue

            ic(f"_worker: Processing {len(individuals)} individuals")
            ic(f"Evaluation worker received {len(individuals)} individuals")

            for individual in individuals:
                if self.stop_event.is_set():
                    ic("_worker: Stop event set, breaking from evaluation loop")
                    break

                try:
                    ic("_worker: Starting evaluation for individual")
                    individual.timer_start()
                    fitness = self.evaluation_function(individual)
                    individual.timer_end()
                    ic(f"_worker: Individual evaluated, fitness: {fitness}, elapse: {individual.get_elapse()}")

                    individual.set_fitness(fitness)
                    individual.set_calculation_state(CalculationState.FINISHED)

                    throughput = 1 / (individual.get_elapse() + 1e-10)
                    ic(f"_worker: Individual throughput: {throughput}")
                    self.response_queue.put(
                        ClientCalculationState(
                            idle=False,
                            throughput=throughput
                        )
                    )

                    ic(f"Evaluated individual with fitness: {fitness}")

                except Exception as e:
                    ic("_worker: Error during evaluation:", e)
                    logging.error(f"Error during evaluation function execution: {e}")
                    individual.set_fitness(float('inf'))
                    continue

            ic("_worker: Finished evaluating all individuals, setting idle state")
            self.response_queue.put(
                ClientCalculationState(
                    idle=True,
                    individuals=individuals,
                )
            )

        ic("_worker: Evaluation worker stopped")
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
            ic("_heartbeat: Socket not alive")
            logging.error("Socket is not alive, cannot send heartbeat")
            return CommunicationResult.CONNECTION_ERROR

        time_since_last = time.time() - self.last_heartbeat
        ic(f"_heartbeat: Time since last heartbeat: {time_since_last:.2f}s")
        if time.time() - self.last_heartbeat < self.heartbeat_interval:
            ic("_heartbeat: Heartbeat not needed yet")
            return CommunicationResult.SUCCESS

        ic(f"_heartbeat: Sending heartbeat with throughput: {throughput}")
        packet = Packet(_packet_type=PacketType.HEARTBEAT, data=throughput)
        result, packet = communicate(self.sock, packet)
        ic("_heartbeat: Heartbeat result:", result)

        if result != CommunicationResult.SUCCESS:
            ic("_heartbeat: Failed to send heartbeat")
            logging.error("Failed to send heartbeat packet")
            return result

        self.last_heartbeat = time.time()
        ic("_heartbeat: Heartbeat sent successfully")

        return CommunicationResult.SUCCESS

    def _request(self) -> CommunicationResult:
        if not self.is_alive():
            ic("_request: Socket not alive")
            logging.error("Socket is not alive, cannot send request")
            return CommunicationResult.CONNECTION_ERROR

        if self.is_assigned():
            ic("_request: Already assigned, skipping request")
            logging.error("Already assigned, cannot send request")
            return CommunicationResult.SUCCESS

        ic("_request: Sending request packet to server")
        packet = Packet(_packet_type=PacketType.REQUEST, data=None)
        result, packet = communicate(self.sock, packet)
        ic("Sending request packet to server")
        ic("_request: Request result:", result, "response:", packet)

        if result != CommunicationResult.SUCCESS:
            ic("_request: Failed to send request")
            logging.error("Failed to send request packet")
            return result

        self.task = packet.data
        ic(f"_request: Received task with {len(self.task) if self.task else 0} individuals")

        return CommunicationResult.SUCCESS

    def _return(self) -> CommunicationResult:
        if not self.is_alive():
            ic("_return: Socket not alive")
            logging.error("Socket is not alive, cannot send individuals")
            return CommunicationResult.CONNECTION_ERROR

        if not self.is_assigned():
            ic("_return: No individuals to return")
            logging.error("No individuals to return, cannot send response")
            return CommunicationResult.SUCCESS

        ic(f"_return: Sending {len(self.evaluated_task)} evaluated individuals to server")
        packet = Packet(_packet_type=PacketType.RESPONSE, data=self.evaluated_task)
        result, packet = communicate(self.sock, packet)
        ic("_return: Return result:", result)

        if result != CommunicationResult.SUCCESS:
            ic("_return: Failed to send response")
            logging.error("Failed to send response packet with individuals")
            return result

        ic(f"Sent {len(self.evaluated_task)} individuals to server")
        ic("_return: Successfully sent individuals, clearing evaluated_task")
        self.evaluated_task = None

        return CommunicationResult.SUCCESS

    def set_throughput(self, throughput: float) -> None:
        self.throughput = throughput

    def run(self) -> Optional[list[Individual]]:
        if not self.is_alive():
            ic("run: Communication worker not alive")
            raise RuntimeError("Communication worker is not running")

        ic(f"run: Running communication worker, throughput: {self.throughput}")
        self._heartbeat(self.throughput)
        if not self.is_assigned():
            ic("run: Not assigned, requesting new task")
            self._request()
        else:
            ic(f"run: Already assigned, task: {self.task is not None}, evaluated_task: {self.evaluated_task is not None}")

        if self.task is None and self.evaluated_task is not None:
            ic("run: Returning evaluated task")
            self._return()

        ic(f"run: Returning task: {self.task is not None}")
        return self.task


def connect_to_server(
        server_address: str,
        port: int,
        evaluation_function: EvaluationFunction,
        handler: Optional[Callable] = None
) -> None:
    ic(f"connect_to_server: Starting client connection to {server_address}:{port}")
    stop_event = threading.Event()

    def signal_handler(signum, frame):
        ic("connect_to_server: Received signal", signum)
        logging.info(f"Received signal {signum}, stopping client...")
        stop_event.set()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    sock = _connect_to_server(server_address, port)

    ic("connect_to_server: Starting evaluation worker")
    evaluation_worker = _EvaluationWorker(evaluation_function)
    evaluation_worker.run()

    ic("connect_to_server: Starting communication worker")
    communication_worker = _CommunicationWorker(sock)

    try:
        iteration = 0
        while not stop_event.is_set():
            iteration += 1
            ic(f"connect_to_server: Main loop iteration {iteration}")
            # Update throughput from evaluation worker
            calc_state = evaluation_worker.get_response()
            ic(f"connect_to_server: Calc state - idle: {calc_state.idle}, throughput: {calc_state.throughput}, individuals: {calc_state.individuals is not None}")
            if calc_state.throughput is not None:
                ic(f"connect_to_server: Setting throughput: {calc_state.throughput}")
                communication_worker.set_throughput(calc_state.throughput)

            if calc_state.individuals is not None:
                ic("connect_to_server: Setting evaluated task")
                communication_worker.set_evaluated_task(calc_state.individuals)

            # Run communication worker and get new task
            try:
                new_task = communication_worker.run()
                if new_task is not None:
                    ic(f"connect_to_server: Received new task with {len(new_task)} individuals")
                    evaluation_worker.add_task(new_task)
                else:
                    ic("connect_to_server: No new task received")
            except Exception as e:
                ic("connect_to_server: Communication error:", e)
                logging.error(f"Communication error: {e}")
                break

            time.sleep(0.01)

    finally:
        ic("connect_to_server: Cleaning up resources")
        # Clean up resources
        evaluation_worker.stop()
        if sock:
            try:
                ic("connect_to_server: Sending disconnection packet")
                disconnect_packet = Packet(_packet_type=PacketType.DISCONNECTION, data=None)
                send_packet(sock, disconnect_packet)
                sock.close()
                ic("connect_to_server: Socket closed")
            except Exception as e:
                ic("connect_to_server: Error during disconnection:", e)
                logging.error(f"Error during disconnection: {e}")
        ic("connect_to_server: Client disconnected")
        logging.info("Client disconnected")
