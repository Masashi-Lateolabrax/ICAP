import dataclasses
import multiprocessing
import time
import signal
from typing import Dict, List, Callable, Optional

import numpy as np
from scipy.optimize import curve_fit

from ..prelude import *
from ._client import connect_to_server


class ThroughputModel:
    def __init__(self):
        self.k = 1.0
        self.alpha = 0.0
        self.observations = {}

    @staticmethod
    def _model(n, k, alpha):
        return (k * n) / (1 + alpha * n)

    def predict(self, num_processes: int) -> float:
        if num_processes <= 0:
            return 0.0
        return self._model(num_processes, self.k, self.alpha)

    def add_observation(self, num_processes: int, throughput: float):
        self.observations[num_processes] = throughput
        self._update_parameters()

    def _update_parameters(self):
        if len(self.observations) < 3:
            return

        n_values = np.array(list(self.observations.keys()))
        t_values = np.array(list(self.observations.values()))

        valid_mask = t_values > 0
        n_valid = n_values[valid_mask]
        t_valid = t_values[valid_mask]

        if len(n_valid) < 2:
            return

        try:
            popt, _ = curve_fit(
                self._model, n_valid, t_valid,
                p0=[self.k, self.alpha],
                bounds=([0.1, 0.0], [10.0, 1.0])
            )
            self.k = 0.7 * self.k + 0.3 * popt[0]
            self.alpha = 0.7 * self.alpha + 0.3 * popt[1]
        except (RuntimeError, ValueError):
            pass

    def find_optimal_count(self, min_count: int, max_count: int) -> int:
        if len(self.observations) < 3:
            return min_count

        best_count = min_count
        best_throughput = 0.0

        for n in range(min_count, max_count + 1):
            predicted = self.predict(n)
            if predicted > best_throughput:
                best_throughput = predicted
                best_count = n
            elif predicted < best_throughput * 0.95:
                break

        return best_count


@dataclasses.dataclass
class ProcessInfo:
    process: multiprocessing.Process
    throughput: Optional[float]
    update_time: Optional[float]

    @property
    def pid(self) -> int:
        return self.process.pid if self.process else None

    @property
    def is_alive(self) -> bool:
        return self.process.is_alive() if self.process else False


class ProcessManager:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.processes: List[multiprocessing.Process] = []
        self.process_throughput: Dict[int, float] = {}
        self.max_processes = multiprocessing.cpu_count()
        self.min_processes = 1

    def get_total_throughput(self) -> float:
        return sum(self.process_throughput.values())

    def update_throughput(self, metrics: ProcessMetrics):
        if not metrics.process_id in self.process_throughput:
            raise RuntimeError(f"Process ID {metrics.process_id} not found in registered processes")
        self.process_throughput[metrics.process_id] = metrics.speed

        print(metrics.format_log_message())

    def _create_client_process(self, evaluation_function: Callable) -> multiprocessing.Process:
        def client_worker():
            try:
                connect_to_server(
                    self.host,
                    self.port,
                    evaluation_function=evaluation_function,
                    handler=self.update_throughput
                )
            except Exception as e:
                print(f"Client process error: {e}")

        return multiprocessing.Process(target=client_worker)

    def adjust_process_count(self, target_count: int, evaluation_function: Callable):
        target_count = max(self.min_processes, min(self.max_processes, target_count))

        self.processes = [p for p in self.processes if p.is_alive()]
        current_count = len(self.processes)

        if target_count > current_count:
            for _ in range(target_count - current_count):
                process = self._create_client_process(evaluation_function)
                process.start()
                self.processes.append(process)
                self.process_throughput[process.pid] = 0.0
        elif target_count < current_count:
            for _ in range(current_count - target_count):
                if self.processes:
                    process = self.processes.pop()
                    pid = process.pid
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                    if pid in self.process_throughput:
                        del self.process_throughput[pid]

    def get_process_count(self) -> int:
        return len([p for p in self.processes if p.is_alive()])

    def stop_all(self):
        for process in self.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
        self.processes.clear()


def collect_throughput_observations(
        manager: ProcessManager,
        model: ThroughputModel,
        evaluation_function: Callable,
        min_processes: int,
        max_processes: int,
        adjustment_interval: float
) -> None:
    print("Collecting throughput observations...")
    for count in range(min_processes, max_processes + 1):
        manager.adjust_process_count(count, evaluation_function)
        time.sleep(adjustment_interval)
        throughput = manager.get_total_throughput()
        model.add_observation(count, throughput)
        print(f"Processes: {count}, throughput: {throughput:.2f}")
    print("Throughput observation collection completed.")


def run_adaptive_client_manager(
        host: str,
        port: int,
        evaluation_function: Callable,
        max_processes: Optional[int] = None,
        adjustment_interval: float = 10.0,
        observation_interval: float = 60.0
):
    """
    Run an adaptive client manager that automatically adjusts the number of processes
    based on throughput observations.
    """
    if max_processes is None:
        max_processes = multiprocessing.cpu_count()

    min_processes = 1

    model = ThroughputModel()
    manager = ProcessManager(host, port)

    running = True
    last_adjustment = time.time()

    def signal_handler(_signum, _frame):
        nonlocal running
        print("\nShutting down...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 50)
    print("ADAPTIVE MULTIPROCESSING CLIENT MANAGER")
    print("=" * 50)
    print(f"Server: {host}:{port}")
    print(f"CPU Cores: {max_processes}")
    print(f"Process Range: 1-{max_processes}")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    collect_throughput_observations(
        manager, model, evaluation_function,
        min_processes, max_processes, adjustment_interval
    )
    last_observation = time.time()

    try:
        while running:
            current_time = time.time()

            if current_time - last_observation >= observation_interval * 60:
                collect_throughput_observations(
                    manager, model, evaluation_function,
                    min_processes, max_processes, adjustment_interval
                )
                last_observation = current_time

            if current_time - last_adjustment >= adjustment_interval * 60:
                current_count = manager.get_process_count()
                optimal_count = model.find_optimal_count(min_processes, max_processes)

                if optimal_count != current_count:
                    manager.adjust_process_count(optimal_count, evaluation_function)
                    print(f"Adjusted: {current_count} -> {optimal_count} processes")

                last_adjustment = current_time

            time.sleep(5)

    except KeyboardInterrupt:
        print("\nReceived interrupt signal")

    finally:
        print("Stopping all processes...")
        manager.stop_all()
        print("All processes stopped.")
