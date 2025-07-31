import dataclasses
import socket
import logging
import multiprocessing
import signal
import queue
import time
from typing import Optional, Callable

import numpy as np
from icecream import ic
from ..prelude import *
from ..types.communication import Packet, PacketType
from ._connection_utils import send_packet, communicate

HEARTBEAT_INTERVAL = 20


@dataclasses.dataclass
class ClientTask:
    def __init__(self, individual: Individual, id_: int):
        self.individual = individual
        self._id = id_

    @property
    def id(self) -> int:
        return self._id

    @property
    def state(self) -> CalculationState:
        return self.individual.get_calculation_state()

    def set_fitness(self, fitness: float) -> None:
        self.individual.set_fitness(fitness)

    def set_state(self, state: CalculationState) -> None:
        self.individual.set_calculation_state(state)

    def copy_from(self, other: 'ClientTask') -> None:
        if not isinstance(other, ClientTask):
            raise TypeError("Can only copy from another ClientTask")
        if not self.id == other.id:
            raise ValueError("Cannot copy from a different ClientTask")
        self.individual.copy_info_data_from(other.individual)

    def __repr__(self) -> str:
        return f"ClientTask(id={self.id}, array={self.individual})"


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


def _evaluation_worker_process(
        task_queue: multiprocessing.Queue,
        response_queue: multiprocessing.Queue,
        evaluation_function: EvaluationFunction,
        stop_event: multiprocessing.Event,
) -> None:
    """Worker process for evaluation tasks"""

    # Set up signal handlers in child process
    def signal_handler(signum, frame):
        logging.info(f"Worker process received signal {signum}, stopping...")
        stop_event.set()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    while not stop_event.is_set():
        try:
            task = task_queue.get(timeout=1)

        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Error getting task from queue: {e}")
            continue

        if not isinstance(task, ClientTask):
            raise TypeError("Expected ClientTask object")

        try:
            fitness = evaluation_function(task.individual)
            task.set_fitness(fitness)
            task.set_state(CalculationState.FINISHED)

        except KeyboardInterrupt:
            logging.info("Evaluation interrupted by user")
            stop_event.set()
            break
        except MemoryError:
            logging.error("Out of memory during evaluation")
            stop_event.set()
            break
        except Exception as e:
            logging.error(f"Error during evaluation function execution: {e}")
            stop_event.set()
            break

        try:
            response_queue.put(task)
        except Exception as e:
            logging.error(f"Error putting result to response queue: {e}")
            continue

    print(f"Process {multiprocessing.current_process().pid} stopping... (stop_event:{stop_event.is_set()})")


class _EvaluationWorker:
    def __init__(
            self,
            evaluation_function: EvaluationFunction,
            num_processes: int = 1
    ):
        self.task_queue = multiprocessing.Queue()
        self.response_queue = multiprocessing.Queue()
        self.evaluation_function = evaluation_function
        self.stop_event = multiprocessing.Event()
        self.processes: list[multiprocessing.Process] = []
        self.num_processes = num_processes

    def is_alive(self) -> bool:
        return any(p.is_alive() for p in self.processes)

    def run(self):
        for i in range(self.num_processes):
            process = multiprocessing.Process(
                target=_evaluation_worker_process,
                args=(
                    self.task_queue,
                    self.response_queue,
                    self.evaluation_function,
                    self.stop_event,
                )
            )
            process.daemon = False
            process.start()
            self.processes.append(process)

    def stop(self):
        if self.is_alive():
            logging.info("Stopping evaluation worker processes...")
            self.stop_event.set()

            for process in self.processes:
                process.join(timeout=3.0)
                if process.is_alive():
                    logging.warning(f"Process {process.pid} did not stop gracefully, terminating...")
                    process.terminate()
                    process.join(timeout=2.0)
                    if process.is_alive():
                        logging.error(f"Process {process.pid} still alive after termination, killing...")
                        process.kill()
                        process.join()

            self.processes.clear()
            logging.info("All evaluation worker processes stopped")
        else:
            logging.warning("Evaluation worker is not running, cannot stop it")

    def add_task(self, tasks: list[ClientTask]) -> None:
        if not self.is_alive():
            raise RuntimeError("Evaluation worker is not running")
        ic(len(tasks))
        for task in tasks:
            self.task_queue.put(task)

    def get_response(self) -> list[ClientTask]:
        if not self.is_alive():
            raise RuntimeError("Evaluation worker is not running")
        tasks = []
        while not self.response_queue.empty():
            tasks.append(
                self.response_queue.get_nowait()
            )
        return tasks


class _CommunicationWorker:
    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.last_heartbeat = time.time()
        self.last_request = 0.0
        self._assigned_individuals: Optional[list[Individual]] = None
        self.calculating_task: list[ClientTask] = []

    def is_alive(self) -> bool:
        return self.sock is not None

    def is_assigned(self) -> bool:
        return bool(self._assigned_individuals)

    def calculation_finished(self) -> bool:
        return len(self.calculating_task) == 0

    def set_evaluated_task(self, tasks: list[ClientTask]) -> list[Individual]:
        if not self.is_assigned():
            logging.error("Not assigned to any task, cannot set evaluated tasks")
            return []
        if self.calculation_finished():
            logging.error("All tasks are already finished, cannot set evaluated tasks")
            return []

        moved_tasks = []

        for task in tasks:
            for idx, t in enumerate(self.calculating_task):
                if task.id == t.id:
                    self.calculating_task.pop(idx)
                    t.copy_from(task)
                    moved_tasks.append(t.individual)
                    break

        return moved_tasks

    def _heartbeat(self) -> CommunicationResult:
        if not self.is_alive():
            logging.error("Socket is not alive, cannot send heartbeat")
            return CommunicationResult.CONNECTION_ERROR

        if time.time() - self.last_heartbeat < HEARTBEAT_INTERVAL:
            return CommunicationResult.SUCCESS

        packet = Packet(_packet_type=PacketType.HEARTBEAT)
        result, packet = communicate(self.sock, packet)

        if result != CommunicationResult.SUCCESS:
            logging.error("Failed to send heartbeat packet")
            return result

        self.last_heartbeat = time.time()

        return CommunicationResult.SUCCESS

    def _mut_request(self) -> CommunicationResult:
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

        self.last_request = time.time()

        self._assigned_individuals = packet.data

        if bool(self._assigned_individuals):
            self.calculating_task = [ClientTask(ind, i) for i, ind in enumerate(self._assigned_individuals)]
        else:
            self.calculating_task = []

        ic(len(self._assigned_individuals) if self._assigned_individuals else None)

        return CommunicationResult.SUCCESS

    def _mut_return(self) -> CommunicationResult:
        if not self.is_alive():
            logging.error("Socket is not alive, cannot send individuals")
            return CommunicationResult.CONNECTION_ERROR

        if not self.is_assigned():
            logging.error("No individuals to return, cannot send response")
            return CommunicationResult.SUCCESS

        if not self.calculation_finished():
            logging.error("Some tasks are still being calculated, cannot send response")
            return CommunicationResult.SUCCESS

        packet = Packet(_packet_type=PacketType.RESPONSE, data=self._assigned_individuals)
        result, packet = communicate(self.sock, packet)

        if result != CommunicationResult.SUCCESS:
            logging.error("Failed to send response packet with individuals")
            return result

        self._assigned_individuals = None

        return CommunicationResult.SUCCESS

    def run(self) -> tuple[CommunicationResult, Optional[list[ClientTask]]]:
        if not self.is_alive():
            raise RuntimeError("Communication worker is not running")

        result = ic(self._heartbeat())
        task = None

        if result != CommunicationResult.SUCCESS:
            return result, task

        if not self.is_assigned():
            result = ic(self._mut_request())
            task = self.calculating_task
        elif self.calculation_finished():
            result = ic(self._mut_return())

        return result, task


def connect_to_server(
        server_address: str,
        port: int,
        evaluation_function: EvaluationFunction,
        handler: Optional[Callable[[list[Individual]], None]] = None,
        num_processes: int = 1
) -> None:
    stop_event = multiprocessing.Event()
    sock = _connect_to_server(server_address, port)

    def signal_handler(signum, frame):
        logging.warning(f"Received signal {signum}, stopping client...")
        stop_event.set()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    evaluation_worker = _EvaluationWorker(evaluation_function, num_processes)
    evaluation_worker.run()

    communication_worker = _CommunicationWorker(sock)

    while not stop_event.is_set():
        evaluated_tasks: list[ClientTask] = ic(evaluation_worker.get_response())
        if evaluated_tasks:
            evaluated_inds: list[Individual] = communication_worker.set_evaluated_task(evaluated_tasks)
            if handler:
                handler(evaluated_inds)

        try:
            result, new_task = communication_worker.run()
            if result != CommunicationResult.SUCCESS:
                logging.error(f"Communication error: {result}")
                break

            if bool(new_task):
                evaluation_worker.add_task(new_task)

        except Exception as e:
            logging.error(f"Communication error: {e}")
            break

        time.sleep(1.0)

    logging.info("Shutting down evaluation worker...")
    evaluation_worker.stop()
    if sock:
        try:
            logging.info("Sending disconnection packet...")
            disconnect_packet = Packet(_packet_type=PacketType.DISCONNECTION, data=None)
            send_packet(sock, disconnect_packet)
        except Exception as e:
            logging.warning(f"Error sending disconnection packet: {e}")
        finally:
            try:
                sock.close()
                logging.info("Socket closed successfully")
            except Exception as e:
                logging.warning(f"Error closing socket: {e}")
