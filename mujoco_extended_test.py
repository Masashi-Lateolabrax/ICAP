#!/usr/bin/env python3
"""
Extended MuJoCo Performance Test

This script provides more comprehensive testing with longer simulations 
and detailed resource monitoring to better understand MuJoCo's behavior
under different workload conditions.
"""

import os
import sys
import time
import multiprocessing as mp
import psutil
import numpy as np
import mujoco
import torch
from dataclasses import dataclass
from typing import List, Tuple, Dict
from statistics import mean, stdev
import threading
import gc


@dataclass
class ExtendedMetrics:
    """Extended performance metrics with detailed resource monitoring."""
    process_count: int
    total_time: float
    avg_time_per_process: float
    min_time: float
    max_time: float
    time_stdev: float
    cpu_usage_percent: float
    memory_usage_mb: float
    peak_memory_mb: float
    steps_per_second: float
    memory_growth_mb: float
    
    def __str__(self) -> str:
        return (f"Processes: {self.process_count:2d} | "
                f"Total: {self.total_time:6.2f}s | "
                f"Avg: {self.avg_time_per_process:6.2f}s | "
                f"Range: {self.min_time:5.2f}-{self.max_time:5.2f}s | "
                f"CPU: {self.cpu_usage_percent:5.1f}% | "
                f"Mem: {self.memory_usage_mb:6.1f}MB | "
                f"Peak: {self.peak_memory_mb:6.1f}MB | "
                f"Steps/s: {self.steps_per_second:7.1f}")


class ResourceMonitor:
    """Monitors system resources during execution."""
    
    def __init__(self):
        self.monitoring = False
        self.cpu_samples = []
        self.memory_samples = []
        self.monitor_thread = None
        
    def start(self):
        """Start monitoring resources."""
        self.monitoring = True
        self.cpu_samples = []
        self.memory_samples = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop(self):
        """Stop monitoring resources."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
            
    def _monitor_loop(self):
        """Monitor loop running in separate thread."""
        process = psutil.Process()
        while self.monitoring:
            try:
                cpu = process.cpu_percent()
                memory = process.memory_info().rss / 1024 / 1024  # MB
                self.cpu_samples.append(cpu)
                self.memory_samples.append(memory)
                time.sleep(0.1)
            except psutil.NoSuchProcess:
                break
                
    def get_metrics(self) -> Tuple[float, float, float]:
        """Get average CPU, average memory, and peak memory."""
        avg_cpu = mean(self.cpu_samples) if self.cpu_samples else 0
        avg_memory = mean(self.memory_samples) if self.memory_samples else 0
        peak_memory = max(self.memory_samples) if self.memory_samples else 0
        return avg_cpu, avg_memory, peak_memory


class OptimizedMuJoCoSimulation:
    """Optimized MuJoCo simulation with better tensor handling."""
    
    def __init__(self, num_robots: int = 5, num_food: int = 3):
        self.num_robots = num_robots
        self.num_food = num_food
        self.world_size = 10.0
        self.setup_simulation()
        
        # Pre-allocate numpy arrays for better performance
        self.sensor_input_np = np.zeros((num_robots, 5), dtype=np.float32)
        self.sensor_input_tensor = torch.from_numpy(self.sensor_input_np)
        
    def setup_simulation(self):
        """Create MuJoCo scene (same as before but with optimizations)."""
        # Create MuJoCo specification
        spec = mujoco.MjSpec()
        
        # Basic physics settings
        spec.option.timestep = 0.01
        spec.option.gravity = (0, 0, -9.81)
        
        # Add ground plane
        ground_body = spec.worldbody
        ground_geom = ground_body.add_geom()
        ground_geom.type = mujoco.mjtGeom.mjGEOM_PLANE
        ground_geom.pos = (0, 0, 0)
        ground_geom.size = (self.world_size * 0.5, self.world_size * 0.5, 1)
        ground_geom.rgba = (0.8, 0.8, 0.8, 1.0)
        
        # Add walls
        for i, (x, y) in enumerate([
            (self.world_size * 0.5, 0), (-self.world_size * 0.5, 0),
            (0, self.world_size * 0.5), (0, -self.world_size * 0.5)
        ]):
            wall_body = spec.worldbody.add_body()
            wall_geom = wall_body.add_geom()
            wall_geom.type = mujoco.mjtGeom.mjGEOM_BOX
            wall_geom.pos = (x, y, 0.5)
            wall_geom.size = (0.1, self.world_size * 0.5 if i < 2 else 0.1, 0.5)
            wall_geom.rgba = (0.5, 0.5, 0.5, 1.0)
        
        # Add robots
        self.robot_bodies = []
        for i in range(self.num_robots):
            x = np.random.uniform(-self.world_size * 0.3, self.world_size * 0.3)
            y = np.random.uniform(-self.world_size * 0.3, self.world_size * 0.3)
            
            robot_body = spec.worldbody.add_body()
            robot_body.name = f"robot_{i}"
            robot_body.pos = (x, y, 0.175)
            
            free_joint = robot_body.add_joint()
            free_joint.name = f"robot_free_{i}"
            free_joint.type = mujoco.mjtJoint.mjJNT_FREE
            
            robot_geom = robot_body.add_geom()
            robot_geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
            robot_geom.size = (0.175, 0.175, 0.175)
            robot_geom.rgba = (1, 1, 0, 1)
            robot_geom.mass = 10.0
            
            robot_site = robot_body.add_site()
            robot_site.name = f"robot_site_{i}"
            robot_site.pos = (0, 0, 0)
            robot_site.size = (0.05, 0.05, 0.05)
            
            self.robot_bodies.append(robot_body)
        
        # Add food objects
        self.food_bodies = []
        for i in range(self.num_food):
            x = np.random.uniform(-self.world_size * 0.4, self.world_size * 0.4)
            y = np.random.uniform(-self.world_size * 0.4, self.world_size * 0.4)
            
            food_body = spec.worldbody.add_body()
            food_body.name = f"food_{i}"
            food_body.pos = (x, y, 0.035)
            
            food_geom = food_body.add_geom()
            food_geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            food_geom.size = (0.5, 0.07, 0.07)
            food_geom.rgba = (0, 1, 1, 1)
            food_geom.density = 80.0
            
            food_site = food_body.add_site()
            food_site.name = f"food_site_{i}"
            food_site.pos = (0, 0, 0)
            food_site.size = (0.05, 0.05, 0.05)
            
            self.food_bodies.append(food_body)
        
        # Add nest
        nest_body = spec.worldbody.add_body()
        nest_body.name = "nest"
        nest_body.pos = (0, 0, 0.01)
        
        nest_geom = nest_body.add_geom()
        nest_geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        nest_geom.size = (1.0, 0.01, 0.01)
        nest_geom.rgba = (0, 1, 0, 1)
        
        nest_site = nest_body.add_site()
        nest_site.name = "nest_site"
        nest_site.pos = (0, 0, 0)
        nest_site.size = (0.05, 0.05, 0.05)
        
        # Compile the model
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        
        # Initialize neural network controller
        self.controller = torch.nn.Sequential(
            torch.nn.Linear(5, 3),
            torch.nn.Tanhshrink(),
            torch.nn.Linear(3, 2),
            torch.nn.Tanh()
        )
        
    def _get_sensor_input_optimized(self):
        """Optimized sensor input generation using pre-allocated arrays."""
        # Generate random sensor data directly into pre-allocated array
        np.random.uniform(-1, 1, size=(self.num_robots, 5), out=self.sensor_input_np)
        return self.sensor_input_tensor
        
    def _apply_control_optimized(self, control_output: torch.Tensor):
        """Optimized control application."""
        control_np = control_output.numpy()
        
        for i in range(self.num_robots):
            if i < len(self.data.qvel):
                left_wheel = control_np[i, 0]
                right_wheel = control_np[i, 1]
                
                linear_vel = (left_wheel + right_wheel) * 0.5 * 0.1
                angular_vel = (right_wheel - left_wheel) * 0.1
                
                base_idx = i * 6
                if base_idx + 5 < len(self.data.qvel):
                    self.data.qvel[base_idx] = linear_vel
                    self.data.qvel[base_idx + 5] = angular_vel
    
    def step(self):
        """Optimized simulation step."""
        # Get sensor input (optimized)
        sensor_input = self._get_sensor_input_optimized()
        
        # Run neural network controller
        with torch.no_grad():
            control_output = self.controller(sensor_input)
        
        # Apply control (optimized)
        self._apply_control_optimized(control_output)
        
        # Step physics
        mujoco.mj_step(self.model, self.data)
        
        # Simple fitness calculation
        fitness = 0.0
        for i in range(self.num_robots):
            robot_pos = self.data.qpos[i * 7:i * 7 + 3]
            fitness += np.linalg.norm(robot_pos[:2])
        
        return fitness


def run_extended_simulation(process_id: int, num_steps: int, return_dict: dict):
    """Run extended simulation with memory monitoring."""
    try:
        # Force garbage collection at start
        gc.collect()
        
        # Record initial memory
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Create simulation
        sim = OptimizedMuJoCoSimulation(num_robots=5, num_food=3)
        
        # Record start time
        start_time = time.time()
        
        # Run simulation steps
        total_fitness = 0.0
        for step in range(num_steps):
            fitness = sim.step()
            total_fitness += fitness
            
            # Periodic garbage collection for longer runs
            if step % 100 == 0:
                gc.collect()
        
        # Record end time
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Record final memory
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        # Store results
        return_dict[process_id] = {
            'elapsed_time': elapsed_time,
            'total_fitness': total_fitness,
            'steps_completed': num_steps,
            'steps_per_second': num_steps / elapsed_time if elapsed_time > 0 else 0,
            'memory_growth': memory_growth,
            'final_memory': final_memory
        }
        
        # Clean up
        del sim
        gc.collect()
        
    except Exception as e:
        return_dict[process_id] = {
            'error': str(e),
            'elapsed_time': 0,
            'total_fitness': 0,
            'steps_completed': 0,
            'steps_per_second': 0,
            'memory_growth': 0,
            'final_memory': 0
        }


def run_extended_test(num_processes: int, num_steps: int = 2000) -> ExtendedMetrics:
    """Run extended performance test with detailed monitoring."""
    print(f"Running extended test: {num_processes} process(es), {num_steps} steps each...")
    
    # Create multiprocessing manager
    manager = mp.Manager()
    return_dict = manager.dict()
    
    # Start resource monitoring
    monitor = ResourceMonitor()
    monitor.start()
    
    # Record initial memory
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    # Create and start processes
    processes = []
    start_time = time.time()
    
    for i in range(num_processes):
        p = mp.Process(target=run_extended_simulation, args=(i, num_steps, return_dict))
        processes.append(p)
        p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Stop monitoring
    monitor.stop()
    avg_cpu, avg_memory, peak_memory = monitor.get_metrics()
    
    # Collect results and debug
    successful_processes = []
    failed_processes = []
    
    for i in range(num_processes):
        if i in return_dict:
            if 'error' in return_dict[i]:
                failed_processes.append(f"Process {i}: {return_dict[i]['error']}")
            else:
                successful_processes.append(return_dict[i])
        else:
            failed_processes.append(f"Process {i}: No results returned")
    
    # Debug output
    if failed_processes:
        print(f"    Failed processes: {len(failed_processes)}")
        for failure in failed_processes:
            print(f"      {failure}")
    
    if not successful_processes:
        print(f"    ERROR: No successful processes out of {num_processes}")
        return ExtendedMetrics(
            process_count=num_processes,
            total_time=total_time,
            avg_time_per_process=0,
            min_time=0,
            max_time=0,
            time_stdev=0,
            cpu_usage_percent=avg_cpu,
            memory_usage_mb=avg_memory,
            peak_memory_mb=peak_memory,
            steps_per_second=0,
            memory_growth_mb=0
        )
    
    # Calculate detailed metrics
    times = [p['elapsed_time'] for p in successful_processes]
    avg_time = mean(times)
    min_time = min(times)
    max_time = max(times)
    time_stdev = stdev(times) if len(times) > 1 else 0
    
    total_steps_per_second = sum([p['steps_per_second'] for p in successful_processes])
    avg_memory_growth = mean([p['memory_growth'] for p in successful_processes])
    
    return ExtendedMetrics(
        process_count=num_processes,
        total_time=total_time,
        avg_time_per_process=avg_time,
        min_time=min_time,
        max_time=max_time,
        time_stdev=time_stdev,
        cpu_usage_percent=avg_cpu,
        memory_usage_mb=avg_memory,
        peak_memory_mb=peak_memory,
        steps_per_second=total_steps_per_second,
        memory_growth_mb=avg_memory_growth
    )


def main():
    """Main function for extended performance testing."""
    print("Extended MuJoCo Performance Test")
    print("=" * 100)
    print(f"CPU cores: {mp.cpu_count()}")
    print(f"Available memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")
    print()
    
    # Test configurations
    test_configs = [
        (1, 1000),   # Single process, short test
        (2, 1000),   # Two processes, short test
        (4, 1000),   # Four processes, short test
        (1, 3000),   # Single process, long test
        (2, 3000),   # Two processes, long test
        (4, 3000),   # Four processes, long test
    ]
    
    results = {}
    
    for num_processes, num_steps in test_configs:
        if num_processes > mp.cpu_count():
            continue
            
        test_name = f"{num_processes}p_{num_steps}s"
        print(f"Running {test_name}...")
        
        try:
            metrics = run_extended_test(num_processes, num_steps)
            results[test_name] = metrics
            print(f"  {metrics}")
        except Exception as e:
            print(f"  ERROR: {e}")
        
        time.sleep(1)
    
    # Analysis
    print("\n" + "=" * 100)
    print("DETAILED ANALYSIS")
    print("=" * 100)
    
    # Short vs long comparison
    if "1p_1000s" in results and "1p_3000s" in results:
        short = results["1p_1000s"]
        long = results["1p_3000s"]
        
        print("Single Process - Short vs Long Simulation:")
        print(f"  Short (1000 steps): {short.steps_per_second:.1f} steps/s")
        print(f"  Long (3000 steps):  {long.steps_per_second:.1f} steps/s")
        
        if short.steps_per_second > 0:
            print(f"  Performance consistency: {(long.steps_per_second/short.steps_per_second)*100:.1f}%")
        else:
            print("  Performance consistency: Cannot calculate (baseline is 0)")
        print()
    
    # Parallel scaling analysis
    for steps in [1000, 3000]:
        print(f"Parallel Scaling ({steps} steps):")
        baseline_key = f"1p_{steps}s"
        if baseline_key in results:
            baseline = results[baseline_key]
            print(f"  Baseline (1 process): {baseline.steps_per_second:.1f} steps/s")
            
            for processes in [2, 4]:
                test_key = f"{processes}p_{steps}s"
                if test_key in results:
                    test_result = results[test_key]
                    efficiency = (test_result.steps_per_second / (baseline.steps_per_second * processes)) * 100
                    print(f"  {processes} processes: {test_result.steps_per_second:.1f} steps/s ({efficiency:.1f}% efficient)")
                    
                    # Time variance analysis
                    if test_result.time_stdev > 0:
                        cv = (test_result.time_stdev / test_result.avg_time_per_process) * 100
                        print(f"    Time variance: {test_result.min_time:.2f}s - {test_result.max_time:.2f}s (CV: {cv:.1f}%)")
                    
                    # Memory analysis
                    if test_result.memory_growth_mb > 0:
                        print(f"    Memory growth: {test_result.memory_growth_mb:.1f}MB per process")
        print()
    
    # Final recommendations
    print("FINAL RECOMMENDATIONS:")
    print("=" * 100)
    
    # Find optimal configuration
    best_efficiency = 0
    best_config = None
    
    for key, metrics in results.items():
        if "1p_" in key:
            continue
        
        processes = int(key.split("p_")[0])
        steps = int(key.split("s")[0].split("_")[1])
        
        baseline_key = f"1p_{steps}s"
        if baseline_key in results:
            baseline = results[baseline_key]
            efficiency = (metrics.steps_per_second / (baseline.steps_per_second * processes)) * 100
            
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_config = (processes, steps, efficiency)
    
    if best_config:
        processes, steps, efficiency = best_config
        print(f"✓ Best configuration: {processes} processes ({efficiency:.1f}% efficient)")
    else:
        print("✓ Single process appears optimal")
    
    # Performance degradation analysis
    degradation_detected = False
    for key, metrics in results.items():
        if "1p_" in key:
            continue
            
        processes = int(key.split("p_")[0])
        steps = int(key.split("s")[0].split("_")[1])
        
        baseline_key = f"1p_{steps}s"
        if baseline_key in results:
            baseline = results[baseline_key]
            efficiency = (metrics.steps_per_second / (baseline.steps_per_second * processes)) * 100
            
            if efficiency < 70:
                degradation_detected = True
                break
    
    if degradation_detected:
        print("⚠ Significant performance degradation detected with multiple processes")
        print("  • MuJoCo shows resource contention issues")
        print("  • Consider process-based parallelism limitations")
        print("  • May need to optimize MuJoCo configuration or use fewer processes")
    else:
        print("✓ MuJoCo scales reasonably well with multiple processes")
        print("  • Resource contention is minimal")
        print("  • Parallel processing is beneficial")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()