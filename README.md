# Self-Organized Robotic Behavior with Pheromones

Experimental code for training robots to learn self-organized behavior using pheromone mechanisms in simulation environments.

## Environment Setup

This project uses [uv](https://github.com/astral-sh/uv) as the Python package manager. Run the following commands to set up the environment for running experiments:

### CUDA Support
```bash
uv sync --extra cu128
# or 
uv sync --extra cu124
```

### CPU Only
```bash
uv sync --extra cpu
```

Note: The extras are mutually exclusive due to UV conflict resolution.

## Overview

This project explores how robots can develop self-organized behaviors through pheromone-based communication, mimicking natural swarm intelligence patterns found in biological systems.

### Objectives
- Train robots to learn self-organized behavior through collective intelligence
- Evaluate the contribution of pheromone mechanisms to robotic behavior
- Analyze the relationship between environmental parameters and emergent behaviors

### Key Contributions
- Experimental platform for training self-organized robotic behavior
- Evaluation of pheromone impact on collective behavior
- Analysis of environmental and systemic variables in robotic swarms

## Project Structure

- **`bin/`**: Experiment execution files, including preliminary and control experiments
- **`framework/`**: Shared utilities and common code used across experiments
- **`libs/`**: Supporting libraries (optimizer, pheromone, sensors, etc.)
- **`scheme/`**: Experiment results organized by name and timestamp

## Method

### Simulation Environment
The simulation uses MuJoCo with the following robot specifications:
- **Sensors**: Two omni-sensors (for robots and food detection)
- **Actuators**: Two wheels for movement
- **Communication**: Pheromone secretion organ
- **Control**: Artificial neural network controller

#### Omni-Sensor Formula
The omni-sensor aggregates positional information using:

$$
\mathrm{OmniSensor} = \frac{1}{N_\mathrm{O}} \sum_{o \in \mathrm{Objects}}
\frac{1}{d_o + 1}
\begin{bmatrix}
\cos \theta_o \\
\sin \theta_o
\end{bmatrix}
$$

Where:
- *Objects*: Set of sensing targets
- $N_\mathrm{O}$: Number of objects
- $d_o$: Distance between robot and object $o$
- $\theta_o$: Relative angle from robot to object $o$

### Task Description
Robots operate in a rectangular field environment where:
- Food items are randomly distributed
- A nest is positioned at the field center
- Food respawns randomly when collected
- **Goal**: Transport food to the nest efficiently

### Optimization
The system uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for neural network parameter optimization.

## Roadmap

1. **Simulation Development**
   - [x] Core framework implementation
   - [x] Basic environment setup
   - [ ] Framework refactoring for debugging
   - [ ] Loss function recording
   - [ ] Discrete action CMA-ES evaluation

2. **Training Optimization**
   - [ ] Identify successful training configurations
   - [ ] Parameter sensitivity analysis

3. **Behavioral Analysis**
   - [ ] Multi-parameter behavior collection
   - [ ] Organization index definition
   - [ ] Setting-behavior relationship analysis

4. **Model Interpretation**
   - [ ] Explainable model conversion
   - [ ] Pheromone effect analysis

## Development Guidelines

### Git Commit Prefixes
| Prefix | Description |
|--------|-------------|
| `add` | New files, directories, or functions |
| `exp` | Experimental setup |
| `mod` | Interface modifications |
| `doc` | Documentation changes |

### Branch Strategy
| Branch | Purpose | Source | Merge Target |
|--------|---------|---------|--------------|
| `main` | Production code | - | - |
| `develop` | Shared development | `main` | `main`, `scheme/*` |
| `scheme/*` | Experiment-specific | `develop` | `main` |

## References

To be added as research progresses.