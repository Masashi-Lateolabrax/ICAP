import dataclasses
import multiprocessing
import time
import signal
from datetime import datetime
from typing import List, Callable, Optional

import numpy as np
from icecream import ic

from ..prelude import *
from ._client import connect_to_server

MIN_PROCESSES = 1


class ThroughputModel:
    def __init__(self):
        self.observations = {}

    def add_observation(self, num_processes: int, throughput: float):
        self.observations[num_processes] = throughput

    def find_optimal_count(self, min_count: int, max_count: int) -> int:
        if not self.observations:
            return min_count

        best_count = min_count
        best_throughput = 0.0

        for num_processes, throughput in self.observations.items():
            if min_count <= num_processes <= max_count and throughput > best_throughput:
                best_throughput = throughput
                best_count = num_processes

        return best_count


@dataclasses.dataclass
class ProcessInfo:
    process: multiprocessing.Process
    throughput_queue: multiprocessing.Queue
    latest_throughput: float = float("nan")

    def get_throughput(self) -> float:
        if self.throughput_queue.empty():
            return self.latest_throughput

        try:
            speed = self.throughput_queue.get_nowait()
            if speed:
                self.latest_throughput = speed
        except Exception as e:
            ic("Error retrieving throughput:", e)

        return self.latest_throughput

    @property
    def pid(self) -> int:
        return self.process.pid if self.process else None

    @property
    def is_alive(self) -> bool:
        return self.process.is_alive() if self.process else False

    def start(self):
        if not self.process.is_alive():
            self.process.start()
            ic("Process started:", self.pid)

    def terminate(self):
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
            ic("Process terminated:", self.pid)
        else:
            ic("Process is not alive, cannot terminate:", self.pid)


class ProcessManager:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.processes: List[ProcessInfo] = []
        self.max_processes = multiprocessing.cpu_count()

    def get_total_throughput(self) -> float:
        total = 0.0
        for process_info in self.processes:
            if process_info.is_alive:
                total += process_info.get_throughput()
        return total

    def all_throughput_observed(self) -> bool:
        return not any(np.isnan(p.get_throughput()) for p in self.processes if p.is_alive)

    def _create_client_process(
            self,
            evaluation_function: Callable,
    ) -> ProcessInfo:
        throughput_queue = multiprocessing.Queue()

        def handler(individual: Individual):
            throughput_queue.put(1 / (individual.get_elapse() + 1e-10))

        def client_worker():
            try:
                connect_to_server(
                    self.host,
                    self.port,
                    evaluation_function=evaluation_function,
                    handler=handler
                )

            except Exception as e:
                ic("Client process error:", e)

        process = multiprocessing.Process(target=client_worker)

        return ProcessInfo(
            process=process,
            throughput_queue=throughput_queue
        )

    def adjust_process_count(self, target_count: int, evaluation_function: Callable):
        target_count = max(MIN_PROCESSES, min(self.max_processes, target_count))

        self.processes = [p for p in self.processes if p.is_alive]
        current_count = len(self.processes)

        if target_count > current_count:
            for _ in range(target_count - current_count):
                process_info = self._create_client_process(evaluation_function)
                process_info.start()
                self.processes.append(process_info)
        elif target_count < current_count:
            for _ in range(current_count - target_count):
                if self.processes:
                    process_info = self.processes.pop()
                    process_info.terminate()

        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] Adjusted process count: {target_count} (current: {current_count})")

    def get_process_count(self) -> int:
        return len([p for p in self.processes if p.is_alive])

    def stop_all(self):
        for process_info in self.processes:
            process_info.terminate()
        self.processes.clear()


def collect_throughput_observations(
        manager: ProcessManager,
        model: ThroughputModel,
        evaluation_function: Callable,
        max_processes: int,
        interval=1  # seconds
) -> None:
    manager.stop_all()

    for count in range(MIN_PROCESSES, max_processes + 1):
        while not manager.all_throughput_observed() or manager.get_process_count() != count:
            manager.adjust_process_count(count, evaluation_function)
            time.sleep(interval)

        throughput = manager.get_total_throughput()
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] Processes: {count}, Throughput: {throughput:.2f} ind/s")
        model.add_observation(count, throughput)


def run_adaptive_client_manager(
        host: str,
        port: int,
        evaluation_function: Callable,
        max_processes: Optional[int] = None,
        observation_interval: float = 60.0  # minutes
):
    """
    Run an adaptive client manager that automatically adjusts the number of processes
    based on throughput observations.
    """
    if max_processes is None:
        max_processes = multiprocessing.cpu_count()

    model = ThroughputModel()
    manager = ProcessManager(host, port)

    running = True

    def signal_handler(_signum, _frame):
        nonlocal running
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{timestamp}] Shutting down...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 50)
    print("ADAPTIVE MULTIPROCESSING CLIENT MANAGER")
    print("=" * 50)
    print("Server:", f"{host}:{port}")
    print("CPU Cores:", max_processes)
    print("Process Range:", f"{MIN_PROCESSES}-{max_processes}")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    collect_throughput_observations(
        manager, model, evaluation_function,
        max_processes
    )
    last_observation = time.time()

    optimal_count = model.find_optimal_count(MIN_PROCESSES, max_processes)

    try:
        while running:
            current_time = time.time()

            manager.adjust_process_count(optimal_count, evaluation_function)

            if current_time - last_observation >= observation_interval * 60:
                collect_throughput_observations(
                    manager, model, evaluation_function,
                    max_processes
                )
                optimal_count = model.find_optimal_count(MIN_PROCESSES, max_processes)
                last_observation = current_time

            time.sleep(60)

    except KeyboardInterrupt:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{timestamp}] Received interrupt signal")

    finally:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] Stopping all processes...")
        manager.stop_all()
        print(f"[{timestamp}] All processes stopped.")
