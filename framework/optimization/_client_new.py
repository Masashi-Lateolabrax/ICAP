import dataclasses
import socket
import logging
import threading
import signal
import queue
import time
from typing import Optional, Callable, Tuple

from ..prelude import *
from ..types.communication import Packet, PacketType
from ._connection_utils import send_packet, receive_packet


@dataclasses.dataclass
class CalculationState:
    idle: bool
    throughput: Optional[float] = None
    individuals: Optional[list[Individual]] = None
    error: Optional[str] = None


def _heartbeat(sock: socket.socket, throughput: float) -> CommunicationResult:
    try:
        packet = Packet(_packet_type=PacketType.HEARTBEAT, data=throughput)
        result = send_packet(sock, packet)
        if result != CommunicationResult.SUCCESS:
            logging.error("Failed to send heartbeat packet")
            return result
        logging.debug(f"Heartbeat sent with throughput: {throughput}")
    except socket.error as e:
        logging.error(f"Socket error during heartbeat: {e}")
        return CommunicationResult.CONNECTION_ERROR

    try:
        result, ack_packet = receive_packet(sock)
        if result != CommunicationResult.SUCCESS:
            logging.error("Failed to receive heartbeat ACK")
            return result
        if ack_packet is None or ack_packet.packet_type != PacketType.ACK:
            logging.error("Received invalid ACK for heartbeat")
            return CommunicationResult.CONNECTION_ERROR
        logging.debug("Heartbeat ACK received successfully")
    except socket.error as e:
        logging.error(f"Socket error during heartbeat ACK: {e}")
        return CommunicationResult.CONNECTION_ERROR

    return CommunicationResult.SUCCESS


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


def _evaluation_worker(
        task_queue: queue.Queue,
        response_queue: queue.Queue,
        evaluation_function: EvaluationFunction,
        stop_event: threading.Event
) -> None:
    while not stop_event.is_set():
        try:
            individuals = task_queue.get(timeout=1.0)
            if individuals is None:
                logging.info("Received poison pill, stopping evaluation worker")
                break
            response_queue.put(
                CalculationState(idle=False, throughput=None)
            )

        except queue.Empty:
            response_queue.put(
                CalculationState(idle=True)
            )
            continue

        logging.debug(f"Evaluation worker received {len(individuals)} individuals")

        for individual in individuals:
            if stop_event.is_set():
                break

            try:
                individual.timer_start()
                fitness = evaluation_function(individual)
                individual.timer_end()

                individual.set_fitness(fitness)
                individual.set_calculation_state(CalculationState.FINISHED)

                response_queue.put(
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

        response_queue.put(
            CalculationState(
                idle=True,
                individuals=individuals,
            )
        )

    logging.info("Evaluation worker stopped")


def _spawn_evaluation_worker(
        evaluation_function: EvaluationFunction,
        stop_event: threading.Event
) -> tuple[threading.Thread, queue.Queue, queue.Queue]:
    task_queue = queue.Queue()
    response_queue = queue.Queue()

    worker_thread = threading.Thread(
        target=_evaluation_worker,
        args=(task_queue, response_queue, evaluation_function, stop_event)
    )
    worker_thread.daemon = True
    worker_thread.start()

    return worker_thread, task_queue, response_queue


def communicate_with_server(
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
    worker_thread, task_queue, response_queue = _spawn_evaluation_worker(evaluation_function, stop_event)

    last_heartbeat = time.time()
    throughput = 0.0

    try:
        while not stop_event.is_set():
            try:
                if time.time() - last_heartbeat > 1.0:
                    if not _send_heartbeat(sock, throughput):
                        logging.error("Failed to send heartbeat")
                        break
                    last_heartbeat = time.time()

                success, packet = receive_packet(sock, retry=3)
                
                if not success:
                    time.sleep(0.1)
                    continue
                
                if packet is None:
                    continue
                
                if packet.packet_type == PacketType.RESPONSE and packet.data is not None:
                    individuals = packet.data
                    if isinstance(individuals, list):
                        logging.debug(f"Received {len(individuals)} individuals from server")
                        task_queue.put(individuals)
                
                try:
                    calc_state = response_queue.get_nowait()
                    if calc_state.idle:
                        request_packet = Packet(_packet_type=PacketType.REQUEST, data=None)
                        if send_packet(sock, request_packet) != CommunicationResult.SUCCESS:
                            logging.error("Failed to send request packet")
                            break
                    
                    if calc_state.throughput is not None:
                        throughput = calc_state.throughput
                    
                    if calc_state.individuals is not None:
                        response_packet = Packet(_packet_type=PacketType.RESPONSE, data=calc_state.individuals)
                        if send_packet(sock, response_packet) != CommunicationResult.SUCCESS:
                            logging.error("Failed to send response packet")
                            break
                        logging.debug(f"Sent {len(calc_state.individuals)} evaluated individuals to server")
                
                except queue.Empty:
                    pass
                
                time.sleep(0.01)
                
            except socket.timeout:
                continue
            except Exception as e:
                logging.error(f"Communication error: {e}")
                break
    
    except Exception as e:
        logging.error(f"Client error: {e}")
    finally:
        stop_event.set()
        
        try:
            task_queue.put(None)
        except:
            pass
        
        try:
            disconnect_packet = Packet(_packet_type=PacketType.DISCONNECTION, data=None)
            send_packet(sock, disconnect_packet)
        except:
            pass
        
        try:
            sock.close()
        except:
            pass
        
        worker_thread.join(timeout=5.0)
        logging.info("Client disconnected")


# Backward compatibility
class OptimizationClient:
    def __init__(self, host: str = 'localhost', port: int = 5000):
        self.host = host
        self.port = port

    def connect(self) -> bool:
        return True

    def disconnect(self) -> None:
        pass

    def run_evaluation_loop(self, evaluation_function: EvaluationFunction, handler: Optional[Callable] = None) -> None:
        communicate_with_server(self.host, self.port, evaluation_function, handler)


def _connect_to_server(
        server_address: str,
        port: int,
        evaluation_function: EvaluationFunction,
        handler: Optional[Callable] = None
) -> None:
    """Backward compatibility function."""
    communicate_with_server(server_address, port, evaluation_function, handler)


def _evaluation_worker(
        task_queue: queue.Queue,
        response_queue: queue.Queue,
        evaluation_function: EvaluationFunction,
        stop_event: threading.Event
) -> None:
    while not stop_event.is_set():
        try:
            individuals = task_queue.get(timeout=1.0)
            if individuals is None:
                logging.info("Received poison pill, stopping evaluation worker")
                break
            response_queue.put(
                CalculationState(idle=False, throughput=None)
            )

        except queue.Empty:
            response_queue.put(
                CalculationState(idle=True)
            )
            continue

        logging.debug(f"Evaluation worker received {len(individuals)} individuals")

        start_time = time.time()

        for individual in individuals:
            if stop_event.is_set():
                break

            try:
                individual.timer_start()
                fitness = evaluation_function(individual)
                individual.timer_end()

                individual.set_fitness(fitness)
                individual.set_calculation_state(CalculationState.FINISHED)

                response_queue.put(
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

        response_queue.put(
            CalculationState(
                idle=True,
                individuals=individuals,
            )
        )

    logging.info("Evaluation worker stopped")
