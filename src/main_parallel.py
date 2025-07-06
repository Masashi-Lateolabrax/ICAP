import multiprocessing
import time
import signal
from typing import Dict, List

import numpy as np
from scipy.optimize import curve_fit

from framework.optimization import connect_to_server
from settings import MySettings


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
        if num_processes in self.observations:
            self.observations[num_processes] = 0.8 * self.observations[num_processes] + 0.2 * throughput
        else:
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
            params, _ = curve_fit(
                self._model, n_valid, t_valid,
                p0=[self.k, self.alpha],
                bounds=([0.1, 0.0], [10.0, 1.0])
            )
            self.k = 0.7 * self.k + 0.3 * params[0]
            self.alpha = 0.7 * self.alpha + 0.3 * params[1]
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


class ProcessManager:
    def __init__(self, settings: MySettings):
        self.settings = settings
        self.processes: List[multiprocessing.Process] = []
        self.process_throughput: Dict[int, float] = {}
        self.max_processes = multiprocessing.cpu_count()
        self.min_processes = 1

    def get_total_throughput(self) -> float:
        return sum(self.process_throughput.values())

    def update_throughput(self, process_id: int, throughput: float):
        if not process_id in self.process_throughput:
            raise RuntimeError(f"Process ID {process_id} not found in registered processes")
        self.process_throughput[process_id] = 0.8 * self.process_throughput[process_id] + 0.2 * throughput

    def _create_client_process(self, evaluation_function) -> multiprocessing.Process:
        def client_worker():
            try:
                connect_to_server(
                    self.settings.Server.HOST,
                    self.settings.Server.PORT,
                    evaluation_function=evaluation_function,
                    handler=lambda metrics: self.update_throughput(metrics.process_id, metrics.speed)
                )
            except Exception as e:
                print(f"Client process error: {e}")

        return multiprocessing.Process(target=client_worker)

    def set_process_count(self, target_count: int, evaluation_function):
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


def main():
    settings = MySettings()
    model = ThroughputModel()
    manager = ProcessManager(settings)

    max_processes = multiprocessing.cpu_count()
    min_processes = 2
    initial_processes = max(2, max_processes // 4)

    running = True
    last_adjustment = time.time()
    adjustment_interval = 30.0

    def signal_handler(signum, frame):
        nonlocal running
        print("\nShutting down...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 50)
    print("ADAPTIVE MULTIPROCESSING CLIENT MANAGER")
    print("=" * 50)
    print(f"Server: {settings.Server.HOST}:{settings.Server.PORT}")
    print(f"CPU Cores: {max_processes}")
    print(f"Initial Processes: {initial_processes}")
    print(f"Process Range: {min_processes}-{max_processes}")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    from src.simulator import evaluate_individual
    manager.set_process_count(initial_processes, evaluate_individual)

    try:
        while running:
            current_time = time.time()

            if current_time - last_adjustment >= adjustment_interval:
                current_count = manager.get_process_count()
                throughput = manager.get_total_throughput()

                model.add_observation(current_count, throughput)
                optimal_count = model.find_optimal_count(min_processes, max_processes)

                if optimal_count != current_count:
                    manager.set_process_count(optimal_count, evaluate_individual)
                    print(f"Adjusted: {current_count} -> {optimal_count} processes")

                print(
                    f"Processes: {manager.get_process_count()}, Throughput: {throughput:.2f}, Model: k={model.k:.3f}, α={model.alpha:.3f}")
                last_adjustment = current_time

            time.sleep(5)

    except KeyboardInterrupt:
        print("\nReceived interrupt signal")

    finally:
        print("Stopping all processes...")
        manager.stop_all()
        print("All processes stopped.")


if __name__ == "__main__":
    main()
