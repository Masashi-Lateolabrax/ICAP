import multiprocessing
import psutil
import time
import signal
from queue import Queue, Empty
from typing import Dict, List

import numpy as np
from scipy.optimize import curve_fit

from framework.prelude import *
from framework.optimization import connect_to_server
from settings import MySettings


class ThroughputModel:
    def __init__(self, initial_k: float = 1.0, initial_alpha: float = 0.0):
        self.k = initial_k
        self.alpha = initial_alpha
        self.distribution: dict[int, float] = {}

    @staticmethod
    def _model(n, k, alpha):
        return (k * n) / (1 + alpha * n)

    def predict(self, num_processes: int) -> float:
        if num_processes <= 0:
            return 0.0
        return ThroughputModel._model(num_processes, self.k, self.alpha)

    def add_observation(self, num_processes: int, throughput: float):
        a = self.distribution[num_processes] if num_processes in self.distribution else 0.0
        self.distribution[num_processes] = 0.8 * a + 0.2 * throughput
        self._update_parameters()

    def _update_parameters(self):
        if len(self.distribution) < 3:
            return

        n_values = np.array(list(self.distribution.keys()))
        t_values = np.array(list(self.distribution.values()))

        # Filter out invalid data points
        valid_mask = t_values > 0
        n_valid = n_values[valid_mask]
        t_valid = t_values[valid_mask]

        if len(n_valid) < 2:
            return

        try:
            fit_result = curve_fit(
                ThroughputModel._model,
                n_valid,
                t_valid,
                p0=np.array([self.k, self.alpha]),
                bounds=([0.1, 0.0], [10.0, 1.0]),  # k > 0.1, 0 <= alpha <= 1
                method='trf'  # Trust Region Reflective algorithm supports bounds
            )

            params_array = np.asarray(fit_result[0])
            new_k = float(params_array[0])
            new_alpha = float(params_array[1])

            self.k = 0.7 * self.k + 0.3 * new_k
            self.alpha = 0.7 * self.alpha + 0.3 * new_alpha

        except (RuntimeError, ValueError):
            pass

    def has_sufficient_data(self) -> bool:
        """Check if we have enough data points for model-based optimization"""
        return len(self.distribution) >= 3

    def find_optimal_process_count(self, min_processes: int, max_processes: int) -> int:
        """Find optimal process count based on saturation model"""
        if not self.has_sufficient_data():
            return min_processes

        best_count = min_processes
        best_throughput = 0.0

        for n in range(min_processes, max_processes + 1):
            predicted_throughput = self.predict(n)
            if predicted_throughput > best_throughput:
                best_throughput = predicted_throughput
                best_count = n
            elif predicted_throughput < best_throughput * 0.95:  # 5% tolerance
                break

        return best_count


class ProcessManager:
    def __init__(self, settings: MySettings):
        self.settings = settings
        self.processes: List[multiprocessing.Process] = []
        self.process_throughput: Dict[int, float] = {}

        self.max_processes = multiprocessing.cpu_count()
        self.min_processes = 1

    @property
    def throughput(self) -> float:
        return sum(self.process_throughput.values())

    def update_process_metrics(self, metrics: ProcessMetrics):
        if metrics.process_id not in self.process_throughput:
            self.process_throughput[metrics.process_id] = 0.0
        a = self.process_throughput[metrics.process_id]
        self.process_throughput[metrics.process_id] = 0.8 * a + 0.2 * metrics.speed

    def _create_client_process(self, evaluation_function) -> multiprocessing.Process:
        def client_worker():
            try:
                connect_to_server(
                    self.settings.Server.HOST,
                    self.settings.Server.PORT,
                    evaluation_function=evaluation_function,
                    handler=self.update_process_metrics
                )
            except Exception as e:
                print(f"Client process error: {e}")

        process = multiprocessing.Process(target=client_worker)
        return process

    def _start_processes(self, count: int, evaluation_function):
        for _ in range(count):
            if len(self.processes) < self.max_processes:
                process = self._create_client_process(evaluation_function)
                process.start()
                self.processes.append(process)

    def _stop_processes(self, count: int):
        stopped = 0
        for i in range(len(self.processes) - 1, -1, -1):
            if stopped >= count:
                break

            process = self.processes[i]
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
                self.processes.pop(i)
                stopped += 1

    def stop_all_processes(self):
        self._stop_processes(len(self.processes))

    def set_num_process(self, n: int, evaluation_function):
        target_processes = max(self.min_processes, n)
        target_processes = min(self.max_processes, target_processes)

        self.processes = [p for p in self.processes if p.is_alive()]
        current_count = len(self.processes)

        if target_processes > current_count:
            self._start_processes(target_processes - current_count, evaluation_function)
        elif target_processes < current_count:
            self._stop_processes(current_count - target_processes)

    def get_process_count(self) -> int:
        return len(self.processes)


class PerformanceMonitor:
    def __init__(self):
        self.metrics_queue = Queue()
        self.process_metrics: Dict[int, ProcessMetrics] = {}
        self.history: List[float] = []
        self.last_adjustment = time.time()
        self.adjustment_interval = 30.0  # seconds
        
    def add_metrics(self, metrics: ProcessMetrics):
        """Add performance metrics from a client process"""
        self.metrics_queue.put(metrics)
        
    def update_metrics(self):
        """Update internal metrics from queue"""
        try:
            while True:
                metrics = self.metrics_queue.get_nowait()
                self.process_metrics[metrics.process_id] = metrics
        except Empty:
            pass
            
    def get_average_speed(self) -> float:
        """Calculate average speed across all processes"""
        if not self.process_metrics:
            return 0.0
            
        # Only consider recent metrics
        recent_metrics = [
            m for m in self.process_metrics.values()
            if m.is_recent()
        ]
        
        if not recent_metrics:
            return 0.0
            
        return sum(m.speed for m in recent_metrics) / len(recent_metrics)
        
    def should_adjust(self) -> bool:
        """Check if we should adjust process count"""
        current_time = time.time()
        return current_time - self.last_adjustment >= self.adjustment_interval
        
    def record_adjustment(self, avg_speed: float):
        """Record an adjustment for history"""
        self.history.append(avg_speed)
        self.last_adjustment = time.time()
        # Keep only last 10 measurements
        if len(self.history) > 10:
            self.history.pop(0)
            
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        return psutil.cpu_percent(interval=1)
        
    def get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        return psutil.virtual_memory().percent
        
    def display_status(self, num_processes: int):
        """Display current status"""
        avg_speed = self.get_average_speed()
        cpu_usage = self.get_cpu_usage()
        memory_usage = self.get_memory_usage()
        
        print(f"\n=== Performance Monitor ===")
        print(f"Active Processes: {num_processes}")
        print(f"Average Speed: {avg_speed:.2f} ind/sec")
        print(f"CPU Usage: {cpu_usage:.1f}%")
        print(f"Memory Usage: {memory_usage:.1f}%")
        print(f"Active Clients: {len(self.process_metrics)}")
        print("=" * 30)


class AdaptiveProcessManager:
    def __init__(self, settings: MySettings):
        self.settings = settings
        self.monitor = PerformanceMonitor()
        self.processes: List[multiprocessing.Process] = []
        self.running = True
        
        # Starting configuration
        self.max_processes = multiprocessing.cpu_count()
        self.min_processes = 2
        self.initial_processes = max(2, self.max_processes // 4)
        self.current_processes = self.initial_processes
        
        # Performance thresholds
        self.cpu_threshold_high = 85.0
        self.cpu_threshold_low = 60.0
        self.speed_improvement_threshold = 0.1  # 10% improvement
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\nShutting down adaptive process manager...")
        self.running = False
        
    def evaluation_function(self, individual: Individual):
        """Same evaluation function as main.py"""
        RobotValues.set_max_speed(self.settings.Robot.MAX_SPEED)
        RobotValues.set_distance_between_wheels(self.settings.Robot.DISTANCE_BETWEEN_WHEELS)
        RobotValues.set_robot_height(self.settings.Robot.HEIGHT)

        backend = Simulator(self.settings, individual)
        for _ in range(math.ceil(self.settings.Simulation.TIME_LENGTH / self.settings.Simulation.TIME_STEP)):
            backend.step()

        return backend.total_score()
        
    def create_client_process(self) -> multiprocessing.Process:
        """Create a new client process"""
        def client_worker():
            try:
                connect_to_server(
                    self.settings.Server.HOST,
                    self.settings.Server.PORT,
                    evaluation_function=self.evaluation_function,
                    handler=self.monitor.add_metrics
                )
            except Exception as e:
                print(f"Client process error: {e}")
                
        process = multiprocessing.Process(target=client_worker)
        return process
        
    def start_processes(self, count: int):
        """Start specified number of processes"""
        for _ in range(count):
            if len(self.processes) < self.max_processes:
                process = self.create_client_process()
                process.start()
                self.processes.append(process)
                
    def stop_processes(self, count: int):
        """Stop specified number of processes"""
        stopped = 0
        for i in range(len(self.processes) - 1, -1, -1):
            if stopped >= count:
                break
                
            process = self.processes[i]
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
                self.processes.pop(i)
                stopped += 1
                
    def cleanup_dead_processes(self):
        """Remove dead processes from the list"""
        self.processes = [p for p in self.processes if p.is_alive()]
        
    def adjust_process_count(self):
        """Adjust process count based on performance"""
        if not self.monitor.should_adjust():
            return
            
        self.monitor.update_metrics()
        avg_speed = self.monitor.get_average_speed()
        cpu_usage = self.monitor.get_cpu_usage()
        memory_usage = self.monitor.get_memory_usage()
        
        # Record current performance
        self.monitor.record_adjustment(avg_speed)
        
        # Decision logic
        current_count = len(self.processes)
        new_count = current_count
        
        # Check if we should increase processes
        if (cpu_usage < self.cpu_threshold_low and 
            memory_usage < 80.0 and 
            current_count < self.max_processes):
            new_count = min(current_count + 1, self.max_processes)
            print(f"Increasing processes: {current_count} -> {new_count} (CPU: {cpu_usage:.1f}%)")
            
        # Check if we should decrease processes
        elif (cpu_usage > self.cpu_threshold_high or 
              memory_usage > 90.0 or
              self._performance_degraded()):
            new_count = max(current_count - 1, self.min_processes)
            print(f"Decreasing processes: {current_count} -> {new_count} (CPU: {cpu_usage:.1f}%)")
            
        # Apply changes
        if new_count > current_count:
            self.start_processes(new_count - current_count)
        elif new_count < current_count:
            self.stop_processes(current_count - new_count)
            
    def _performance_degraded(self) -> bool:
        """Check if performance has degraded"""
        if len(self.monitor.history) < 3:
            return False
            
        # Compare recent performance with previous
        recent_avg = sum(self.monitor.history[-2:]) / 2
        older_avg = sum(self.monitor.history[-4:-2]) / 2
        
        return recent_avg < older_avg * (1 - self.speed_improvement_threshold)
        
    def run(self):
        """Main execution loop"""
        print("=" * 60)
        print("ADAPTIVE MULTIPROCESSING CLIENT MANAGER")
        print("=" * 60)
        print(f"Server: {self.settings.Server.HOST}:{self.settings.Server.PORT}")
        print(f"CPU Cores: {multiprocessing.cpu_count()}")
        print(f"Initial Processes: {self.initial_processes}")
        print(f"Process Range: {self.min_processes}-{self.max_processes}")
        print("-" * 60)
        print("Starting initial processes...")
        print("Press Ctrl+C to stop")
        print("=" * 60)
        
        # Start initial processes
        self.start_processes(self.initial_processes)
        
        try:
            while self.running:
                self.cleanup_dead_processes()
                self.adjust_process_count()
                self.monitor.update_metrics()
                self.monitor.display_status(len(self.processes))
                
                time.sleep(5)  # Check every 5 seconds
                
        except KeyboardInterrupt:
            print("\nReceived interrupt signal")
            
        finally:
            print("Stopping all processes...")
            self.stop_processes(len(self.processes))
            print("All processes stopped.")


def main():
    settings = MySettings()
    manager = AdaptiveProcessManager(settings)
    manager.run()


if __name__ == "__main__":
    main()