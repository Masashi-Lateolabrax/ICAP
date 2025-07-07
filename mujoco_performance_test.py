#!/usr/bin/env python3
"""
MuJoCo Performance Test Script

This script measures MuJoCo's actual resource usage and performance when running
multiple instances in parallel vs single instance. It creates a minimal simulation
similar to the ICAP project setup and compares performance across different
process counts.

The goal is to empirically verify whether MuJoCo really has resource contention
issues when running multiple instances in parallel.
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
from typing import List, Tuple
from statistics import mean, stdev


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results."""
    process_count: int
    total_time: float
    avg_time_per_process: float
    cpu_usage_percent: float
    memory_usage_mb: float
    steps_per_second: float
    
    def __str__(self) -> str:
        return (f"Processes: {self.process_count:2d} | "
                f"Total Time: {self.total_time:6.2f}s | "
                f"Avg/Process: {self.avg_time_per_process:6.2f}s | "
                f"CPU: {self.cpu_usage_percent:5.1f}% | "
                f"Memory: {self.memory_usage_mb:6.1f}MB | "
                f"Steps/s: {self.steps_per_second:6.1f}")


class MinimalMuJoCoSimulation:
    """Minimal MuJoCo simulation mimicking ICAP project setup."""
    
    def __init__(self, num_robots: int = 5, num_food: int = 3):
        self.num_robots = num_robots
        self.num_food = num_food
        self.world_size = 10.0
        self.setup_simulation()
        
    def setup_simulation(self):
        """Create a minimal MuJoCo scene with robots and food objects."""
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
        
        # Add robots (simplified as spheres with actuators)
        self.robot_bodies = []
        for i in range(self.num_robots):
            # Random position
            x = np.random.uniform(-self.world_size * 0.3, self.world_size * 0.3)
            y = np.random.uniform(-self.world_size * 0.3, self.world_size * 0.3)
            
            # Robot body
            robot_body = spec.worldbody.add_body()
            robot_body.name = f"robot_{i}"
            robot_body.pos = (x, y, 0.175)
            
            # Add free joint for robot movement
            free_joint = robot_body.add_joint()
            free_joint.name = f"robot_free_{i}"
            free_joint.type = mujoco.mjtJoint.mjJNT_FREE
            
            # Robot geometry
            robot_geom = robot_body.add_geom()
            robot_geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
            robot_geom.size = (0.175, 0.175, 0.175)
            robot_geom.rgba = (1, 1, 0, 1)
            robot_geom.mass = 10.0
            
            # Add site for sensors
            robot_site = robot_body.add_site()
            robot_site.name = f"robot_site_{i}"
            robot_site.pos = (0, 0, 0)
            robot_site.size = (0.05, 0.05, 0.05)
            
            self.robot_bodies.append(robot_body)
        
        # Add food objects
        self.food_bodies = []
        for i in range(self.num_food):
            # Random position
            x = np.random.uniform(-self.world_size * 0.4, self.world_size * 0.4)
            y = np.random.uniform(-self.world_size * 0.4, self.world_size * 0.4)
            
            # Food body
            food_body = spec.worldbody.add_body()
            food_body.name = f"food_{i}"
            food_body.pos = (x, y, 0.035)
            
            # Food geometry
            food_geom = food_body.add_geom()
            food_geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            food_geom.size = (0.5, 0.07, 0.07)
            food_geom.rgba = (0, 1, 1, 1)
            food_geom.density = 80.0
            
            # Add site for sensors
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
        
        # Initialize neural network controller (simplified)
        self.controller = self._create_simple_controller()
        
    def _create_simple_controller(self) -> torch.nn.Module:
        """Create a simple neural network controller similar to ICAP."""
        return torch.nn.Sequential(
            torch.nn.Linear(5, 3),  # 5 inputs: 2×omni + 1×direction
            torch.nn.Tanhshrink(),
            torch.nn.Linear(3, 2),  # 2 outputs: left/right wheel speeds
            torch.nn.Tanh()
        )
    
    def _get_sensor_input(self) -> torch.Tensor:
        """Generate mock sensor input (simplified)."""
        # In real ICAP, this would be complex sensor calculations
        # Here we just generate some fake sensory data
        inputs = []
        for i in range(self.num_robots):
            # Mock robot sensor (2 values)
            robot_sensor = np.random.uniform(-1, 1, 2)
            # Mock food sensor (2 values)  
            food_sensor = np.random.uniform(-1, 1, 2)
            # Mock direction sensor (1 value)
            direction_sensor = np.random.uniform(-1, 1, 1)
            
            robot_input = np.concatenate([robot_sensor, food_sensor, direction_sensor])
            inputs.append(robot_input)
        
        return torch.tensor(inputs, dtype=torch.float32)
    
    def _apply_control(self, control_output: torch.Tensor):
        """Apply control commands to robots (simplified)."""
        # In real ICAP, this would set actuator values
        # Here we just apply small random forces to simulate robot movement
        for i in range(self.num_robots):
            if i < len(self.data.qvel):
                # Apply small random forces to simulate wheel control
                left_wheel = control_output[i, 0].item()
                right_wheel = control_output[i, 1].item()
                
                # Simplified differential drive model
                linear_vel = (left_wheel + right_wheel) * 0.5 * 0.1
                angular_vel = (right_wheel - left_wheel) * 0.1
                
                # Apply to free joint velocities (simplified)
                base_idx = i * 6  # Each free joint has 6 DOFs
                if base_idx + 5 < len(self.data.qvel):
                    self.data.qvel[base_idx] = linear_vel  # x velocity
                    self.data.qvel[base_idx + 5] = angular_vel  # z angular velocity
    
    def step(self):
        """Perform one simulation step."""
        # Get sensor input
        sensor_input = self._get_sensor_input()
        
        # Run neural network controller
        with torch.no_grad():
            control_output = self.controller(sensor_input)
        
        # Apply control
        self._apply_control(control_output)
        
        # Step physics
        mujoco.mj_step(self.model, self.data)
        
        # Simple fitness calculation (distance-based)
        fitness = 0.0
        for i in range(self.num_robots):
            # Mock fitness based on position
            robot_pos = self.data.qpos[i * 7:i * 7 + 3]  # x, y, z from free joint
            fitness += np.linalg.norm(robot_pos[:2])  # Distance from origin
        
        return fitness


def run_simulation_process(process_id: int, num_steps: int, return_dict: dict):
    """Run a single simulation process."""
    try:
        # Create simulation
        sim = MinimalMuJoCoSimulation(num_robots=5, num_food=3)
        
        # Record start time
        start_time = time.time()
        
        # Run simulation steps
        total_fitness = 0.0
        for step in range(num_steps):
            fitness = sim.step()
            total_fitness += fitness
        
        # Record end time
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Store results
        return_dict[process_id] = {
            'elapsed_time': elapsed_time,
            'total_fitness': total_fitness,
            'steps_completed': num_steps,
            'steps_per_second': num_steps / elapsed_time if elapsed_time > 0 else 0
        }
        
    except Exception as e:
        return_dict[process_id] = {
            'error': str(e),
            'elapsed_time': 0,
            'total_fitness': 0,
            'steps_completed': 0,
            'steps_per_second': 0
        }


def measure_system_resources(duration: float) -> Tuple[float, float]:
    """Measure CPU and memory usage during a time period."""
    process = psutil.Process()
    cpu_samples = []
    memory_samples = []
    
    start_time = time.time()
    while time.time() - start_time < duration:
        cpu_samples.append(process.cpu_percent())
        memory_samples.append(process.memory_info().rss / 1024 / 1024)  # MB
        time.sleep(0.1)
    
    return mean(cpu_samples) if cpu_samples else 0, mean(memory_samples) if memory_samples else 0


def run_performance_test(num_processes: int, num_steps: int = 1000) -> PerformanceMetrics:
    """Run performance test with specified number of processes."""
    print(f"Running test with {num_processes} process(es), {num_steps} steps each...")
    
    # Create multiprocessing manager for returning results
    manager = mp.Manager()
    return_dict = manager.dict()
    
    # Start system resource monitoring
    monitor_start = time.time()
    
    # Create and start processes
    processes = []
    start_time = time.time()
    
    for i in range(num_processes):
        p = mp.Process(target=run_simulation_process, args=(i, num_steps, return_dict))
        processes.append(p)
        p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Collect results
    successful_processes = []
    failed_processes = []
    
    for i in range(num_processes):
        if i in return_dict:
            if 'error' in return_dict[i]:
                failed_processes.append(i)
                print(f"Process {i} failed: {return_dict[i]['error']}")
            else:
                successful_processes.append(return_dict[i])
        else:
            failed_processes.append(i)
            print(f"Process {i} did not return results")
    
    if not successful_processes:
        print(f"ERROR: No processes completed successfully for {num_processes} processes")
        return PerformanceMetrics(
            process_count=num_processes,
            total_time=total_time,
            avg_time_per_process=0,
            cpu_usage_percent=0,
            memory_usage_mb=0,
            steps_per_second=0
        )
    
    # Calculate metrics
    avg_time_per_process = mean([p['elapsed_time'] for p in successful_processes])
    total_steps_per_second = sum([p['steps_per_second'] for p in successful_processes])
    
    # Measure system resources (approximate)
    cpu_usage, memory_usage = measure_system_resources(0.5)
    
    return PerformanceMetrics(
        process_count=num_processes,
        total_time=total_time,
        avg_time_per_process=avg_time_per_process,
        cpu_usage_percent=cpu_usage,
        memory_usage_mb=memory_usage,
        steps_per_second=total_steps_per_second
    )


def main():
    """Main function to run the performance test suite."""
    print("MuJoCo Performance Test")
    print("=" * 80)
    print(f"CPU cores: {mp.cpu_count()}")
    print(f"Available memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")
    print()
    
    # Test configurations
    process_counts = [1, 2, 4, 8]
    num_steps = 1000
    
    results = []
    
    # Run tests
    for num_processes in process_counts:
        # Skip if more processes than CPU cores
        if num_processes > mp.cpu_count():
            print(f"Skipping {num_processes} processes (exceeds CPU count)")
            continue
            
        try:
            metrics = run_performance_test(num_processes, num_steps)
            results.append(metrics)
            print(f"  {metrics}")
        except Exception as e:
            print(f"ERROR: Failed to run test with {num_processes} processes: {e}")
        
        # Small delay between tests
        time.sleep(2)
    
    # Summary
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"{'Processes':<10} {'Total Time':<12} {'Avg/Process':<12} {'Total Steps/s':<12} {'Efficiency':<12}")
    print("-" * 80)
    
    baseline_sps = None
    for metrics in results:
        if metrics.process_count == 1:
            baseline_sps = metrics.steps_per_second
        
        efficiency = ""
        if baseline_sps and baseline_sps > 0:
            theoretical_sps = baseline_sps * metrics.process_count
            actual_efficiency = (metrics.steps_per_second / theoretical_sps) * 100
            efficiency = f"{actual_efficiency:.1f}%"
        
        print(f"{metrics.process_count:<10} {metrics.total_time:<12.2f} {metrics.avg_time_per_process:<12.2f} "
              f"{metrics.steps_per_second:<12.1f} {efficiency:<12}")
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    if len(results) >= 2:
        single_process = results[0]
        multi_process = [r for r in results if r.process_count > 1]
        
        print(f"Single process baseline: {single_process.steps_per_second:.1f} steps/second")
        print()
        
        for mp_result in multi_process:
            theoretical_speedup = mp_result.process_count
            actual_speedup = mp_result.steps_per_second / single_process.steps_per_second
            efficiency = (actual_speedup / theoretical_speedup) * 100
            
            print(f"{mp_result.process_count} processes:")
            print(f"  Theoretical speedup: {theoretical_speedup:.1f}x")
            print(f"  Actual speedup: {actual_speedup:.1f}x")
            print(f"  Parallel efficiency: {efficiency:.1f}%")
            
            if efficiency < 80:
                print(f"  → Significant performance degradation detected!")
            elif efficiency < 90:
                print(f"  → Moderate performance degradation")
            else:
                print(f"  → Good parallel scaling")
            print()
    
    # Recommendations
    print("RECOMMENDATIONS:")
    if results:
        best_efficiency = max(results, key=lambda x: x.steps_per_second / x.process_count)
        print(f"- Optimal process count appears to be: {best_efficiency.process_count}")
        
        if any(r.steps_per_second / r.process_count < results[0].steps_per_second * 0.8 for r in results[1:]):
            print("- MuJoCo shows resource contention with multiple processes")
            print("- Consider using fewer parallel processes or optimizing MuJoCo usage")
        else:
            print("- MuJoCo scales well with multiple processes")
            print("- Resource contention is minimal")


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()