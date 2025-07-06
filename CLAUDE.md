# ICAP Project Notes

## Project Overview
**ICAP (Intelligent Collective Autonomous Phenomenology)** is an experimental robotics research platform that studies self-organized robotic behavior using pheromone-based mechanisms. The project implements a distributed optimization system that trains robots to learn collective behaviors through swarm intelligence, specifically focusing on food foraging and transport tasks.

### Research Goals
- Train robots to develop self-organized behaviors through collective intelligence
- Evaluate pheromone mechanisms' contribution to robotic swarm behavior  
- Analyze relationships between environmental parameters and emergent behaviors
- Create an explainable model for understanding collective robotics

## Architecture & Performance
- **Server**: Single-threaded CMA-ES optimization server
- **Clients**: Multiple processes for distributed fitness evaluation  
- **Communication**: Socket-based with Queue for thread safety
- **CPU Usage**: Use process-based parallelism, not threading
- **SOCKET_BACKLOG**: Controls TCP queue size, not parallel processing
- **Physics**: MuJoCo integration for realistic robot dynamics

## Critical Configuration
- `MySettings.Optimization.dimension = None` (line 34) must remain None - set dynamically based on neural network dimension
- **Population Size**: 1000 (CMA-ES)
- **Sigma**: 0.5 (CMA-ES initial standard deviation)
- **Simulation Time**: 60s per evaluation
- **Timestep**: 0.01s

## Key Files & Structure

### Main Entry Points
- `src/main.py`: Client entry point - connects to server and runs simulations
- `src/server.py`: Optimization server - manages CMA-ES and coordinates clients
- `src/simulator.py`: Main simulation backend using MuJoCo physics engine
- `src/settings.py`: Application-specific configuration overrides

### Core Framework (`framework/`)
- `optimization/_server.py`: Core server implementation
- `optimization/_client.py`: Client connection handling
- `backends/_mujoco.py`: MuJoCo physics backend
- `sensor/`: Robot sensor implementations (omni-sensor, direction sensor)
- `config/_settings.py`: Base framework settings
- `types/`: Data structures and type definitions
- `environment/`: Environment setup and object positioning

### Neural Network Architecture
```python
Sequential(
    Linear(5, 3),      # Input: 2×omni-sensor + 1×direction
    Tanhshrink(),      # Non-linear activation
    Linear(3, 2),      # Output: left/right wheel speeds
    Tanh()            # Bounded output [-1, 1]
)
```

### Sensor System
1. **Robot Omni-Sensor**: Detects other robots using preprocessed omnidirectional sensing
2. **Food Omni-Sensor**: Detects food items with distance-based weighting
3. **Direction Sensor**: Provides bearing to nest location

## Running the System

### Server
```bash
PYTHONPATH=. uv run src/server.py
```

### Client
```bash
PYTHONPATH=. uv run src/main.py
```

### Dependencies
- **Package Manager**: `uv` (modern Python package manager)
- **Physics**: `mujoco==3.3.3`
- **Optimization**: `cmaes>=0.11.1`
- **Neural Networks**: `torch`
- **Hardware**: CUDA 12.4/12.8 support with CPU fallback

## Code Analysis Framework

### Analysis Process
1. **Context**: Why does this code exist? What's its role?
2. **Problem**: What's broken and why?
3. **Solution**: Fix requirements without breaking other parts

### Common Mistakes
- Surface-level judgments (e.g., "time.sleep() → delete")
- Ignoring system architecture
- Repeating same errors after feedback
- Not understanding distributed optimization requirements

### Key Principles
- Ask "why" three times before changing code
- Understand system-wide impact
- Design before implementing
- Admit mistakes and pivot when necessary
- Respect the distributed client-server architecture
- Maintain physics simulation accuracy

## Research Context
This is a sophisticated robotics research platform studying emergent collective behaviors in multi-robot systems. The distributed optimization approach allows for efficient parallel evaluation of neural network controllers while maintaining scientific rigor in the simulation environment.