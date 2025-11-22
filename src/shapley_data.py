"""Data structures for Shapley value calculation."""

from dataclasses import dataclass
import numpy as np


@dataclass
class ShapleyInputSample:
    """Single sample of robot input data for Shapley value calculation.

    This captures robot sensor inputs at a specific timestep when pheromone
    is being detected, for analyzing the contribution of different sensors.
    """
    # Time information
    timestep: int
    time_seconds: float

    # Robot identification
    robot_index: int

    # Full input vector (9 dimensions)
    full_input: np.ndarray  # Shape: (9,)

    # Decomposed sensor inputs
    robot_sensor: np.ndarray       # Shape: (2,) - PreprocessedOmniSensor for robots
    food_sensor: np.ndarray        # Shape: (2,) - PreprocessedOmniSensor for food
    direction_sensor: np.ndarray   # Shape: (2,) - DirectionSensor to nest
    pheromone_magnitude: float     # Scalar - normalized pheromone concentration
    pheromone_grad_forward: float  # Scalar - gradient in forward direction
    pheromone_grad_side: float     # Scalar - gradient in sideways direction

    # Network output for context
    network_output: np.ndarray     # Shape: (3,) - [right_wheel, left_wheel, pheromone_secretion]

    # Additional context
    robot_position: np.ndarray     # Shape: (2,) - (x, y)
    robot_direction: np.ndarray    # Shape: (2,) - unit vector

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'timestep': self.timestep,
            'time_seconds': self.time_seconds,
            'robot_index': self.robot_index,
            'full_input': self.full_input.tolist(),
            'robot_sensor': self.robot_sensor.tolist(),
            'food_sensor': self.food_sensor.tolist(),
            'direction_sensor': self.direction_sensor.tolist(),
            'pheromone_magnitude': self.pheromone_magnitude,
            'pheromone_grad_forward': self.pheromone_grad_forward,
            'pheromone_grad_side': self.pheromone_grad_side,
            'network_output': self.network_output.tolist(),
            'robot_position': self.robot_position.tolist(),
            'robot_direction': self.robot_direction.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ShapleyInputSample':
        """Create from dictionary."""
        return cls(
            timestep=data['timestep'],
            time_seconds=data['time_seconds'],
            robot_index=data['robot_index'],
            full_input=np.array(data['full_input']),
            robot_sensor=np.array(data['robot_sensor']),
            food_sensor=np.array(data['food_sensor']),
            direction_sensor=np.array(data['direction_sensor']),
            pheromone_magnitude=data['pheromone_magnitude'],
            pheromone_grad_forward=data['pheromone_grad_forward'],
            pheromone_grad_side=data['pheromone_grad_side'],
            network_output=np.array(data['network_output']),
            robot_position=np.array(data['robot_position']),
            robot_direction=np.array(data['robot_direction']),
        )


@dataclass
class ShapleyDataset:
    """Collection of Shapley input samples with metadata."""

    # Collection metadata
    experiment_id: str
    generation: int
    individual_id: str
    simulation_duration: float
    pheromone_threshold: float
    collection_timestamp: str

    # Collected samples
    samples: list[ShapleyInputSample]

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'experiment_id': self.experiment_id,
            'generation': self.generation,
            'individual_id': self.individual_id,
            'simulation_duration': self.simulation_duration,
            'pheromone_threshold': self.pheromone_threshold,
            'collection_timestamp': self.collection_timestamp,
            'samples': [s.to_dict() for s in self.samples],
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ShapleyDataset':
        """Create from dictionary."""
        return cls(
            experiment_id=data['experiment_id'],
            generation=data['generation'],
            individual_id=data['individual_id'],
            simulation_duration=data['simulation_duration'],
            pheromone_threshold=data['pheromone_threshold'],
            collection_timestamp=data['collection_timestamp'],
            samples=[ShapleyInputSample.from_dict(s) for s in data['samples']],
        )

    def get_summary(self) -> str:
        """Get summary statistics of the dataset."""
        if len(self.samples) == 0:
            return "Empty dataset"

        robot_counts = {}
        for sample in self.samples:
            robot_counts[sample.robot_index] = robot_counts.get(sample.robot_index, 0) + 1

        pheromone_magnitudes = [s.pheromone_magnitude for s in self.samples]

        summary = f"""Shapley Dataset Summary
=======================
Experiment: {self.experiment_id}
Generation: {self.generation}
Individual: {self.individual_id}
Simulation Duration: {self.simulation_duration:.1f}s
Pheromone Threshold: {self.pheromone_threshold}
Collection Time: {self.collection_timestamp}

Total Samples: {len(self.samples)}
Samples per Robot: {robot_counts}

Pheromone Magnitude Statistics:
  Min: {np.min(pheromone_magnitudes):.4f}
  Max: {np.max(pheromone_magnitudes):.4f}
  Mean: {np.mean(pheromone_magnitudes):.4f}
  Std: {np.std(pheromone_magnitudes):.4f}
"""
        return summary
