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

## Running the System

**Important**: All commands must be executed from the project root directory.

### Environment Setup
```bash
# CUDA 12.8 support
uv sync --extra cu128

# CUDA 12.4 support  
uv sync --extra cu124

# CPU only
uv sync --extra cpu
```

Note: The extras are mutually exclusive due to UV conflict resolution.

### Server
```bash
# For CUDA 12.8 support
PYTHONPATH=. uv run src/server.py --extra cu128

# For CUDA 12.4 support  
PYTHONPATH=. uv run src/server.py --extra cu124

# For CPU-only execution
PYTHONPATH=. uv run src/server.py --extra cpu
```

### Client
```bash
# Primary client for robot behavior evaluation
PYTHONPATH=. uv run src/main.py --extra cu128

# Example client for testing (uses simple Rosenbrock function)
PYTHONPATH=. uv run examples/optimization/client.py --extra cu128
```

Replace `cu128` with `cu124` or `cpu` depending on your environment setup.

### Dependencies
- **Package Manager**: `uv` (modern Python package manager)
- **Physics**: `mujoco==3.3.3`
- **Optimization**: `cmaes>=0.11.1`
- **Neural Networks**: `torch`
- **Hardware**: CUDA 12.4/12.8 support with CPU fallback

### Pillow Compatibility
If you encounter `PIL._imagingtk` errors with uv's standalone Python:

1. Install dependencies: `sudo apt install tk-dev tcl-dev`
2. Install Python via pyenv: `pyenv install 3.12.10`
3. Remove uv's standalone Python: `rm -rf ~/.local/share/uv/python`

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

## Task Description & Method

### Simulation Environment
The simulation uses MuJoCo with the following robot specifications:
- **Sensors**: Two omni-sensors (for robots and food detection)
- **Actuators**: Two wheels for movement
- **Communication**: Pheromone secretion organ
- **Control**: Artificial neural network controller

### Task
Robots operate in a rectangular field environment where:
- Food items are randomly distributed
- A nest is positioned at the field center
- Food respawns randomly when collected
- **Goal**: Transport food to the nest efficiently

### Optimization
The system uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for neural network parameter optimization.

## Development Guidelines

### Git Commit Prefixes
| Prefix | Description                          |
|--------|--------------------------------------|
| `add`  | New files, directories, or functions |
| `exp`  | Experimental setup                   |
| `mod`  | Interface modifications              |
| `doc`  | Documentation changes                |

### Branch Strategy
| Branch     | Purpose             | Source    | Merge Target       |
|------------|---------------------|-----------|-------------------|
| `main`     | Production code     | -         | -                 |
| `develop`  | Shared development  | `main`    | `main`, `scheme/*` |
| `scheme/*` | Experiment-specific | `develop` | `main`            |

## Common Issues & Solutions

### Socket Connection Management
- **Problem**: Socket timeout warnings loop infinitely when clients disconnect
- **Root Cause**: Insufficient socket health checking in `_connection_utils.py`
- **Solution**: Add `getpeername()` checks before socket operations to detect disconnected clients early
- **Files**: `framework/optimization/_connection_utils.py`, `framework/optimization/_connection.py`

### Distribution System Errors
- **Problem**: "Connection not found in batch size tracking" warnings
- **Root Cause**: `_update_batch_size()` clears `batch_size` dictionary but may not repopulate all connections
- **Solution**: Ensure all active connections are always present in `batch_size` tracking
- **Files**: `framework/optimization/_distribution.py`

### Client Disconnection Handling
- **Symptom**: Server continues attempting operations on disconnected clients
- **Solution**: Proper cleanup in `_remove_unhealthy_connection_info()` to remove from both `performance` and `batch_size` dictionaries
- **Critical**: Server must handle client disconnections gracefully without infinite retry loops

### Connection Health Monitoring
- **Timeout Settings**: 1.0s for server connections, 10.0s for client connections
- **Retry Logic**: `ATTEMPT_COUNT = 10` max attempts before marking connection as failed
- **Health Check**: `Connection.is_healthy` uses `getpeername()`, `getsockopt()`, and `select()` for comprehensive validation

## Research Context
This is a sophisticated robotics research platform studying emergent collective behaviors in multi-robot systems. The distributed optimization approach allows for efficient parallel evaluation of neural network controllers while maintaining scientific rigor in the simulation environment.

## Client-Server Communication Enhancement (Completed)

### New Architecture Design
Packet-based client-server communication system implemented:

1. **Multithreaded Client Design**:
   - ✅ 2 threads per client: simulation execution thread + communication thread
   - ✅ Communication on main thread with regular heartbeat and server requests
   - ✅ Simulation execution on separate thread to avoid blocking communication

2. **Packet-Based Communication System**:
   - ✅ `framework/types/communication.py`: Implements structured communication
   - ✅ `PacketType` enum with 6 types:
     - `HANDSHAKE`: Initial connection setup (no data)
     - `HEARTBEAT`: Regular keepalive with processing speed data
     - `REQUEST`: Request for Individuals from server (no data)
     - `RESPONSE`: Send Individuals to server (contains Individual data)
     - `DISCONNECTION`: Notify before disconnecting (no data)
     - `ACK`: Acknowledgment response (may contain data or be empty)
   - ✅ `Packet` dataclass with `packet_type` and `data` fields
   - ✅ `SocketState` class for managing client connection state

3. **Implementation Status**:
   - ✅ PacketType and Packet classes implemented
   - ✅ Client multithreading architecture (framework/optimization/_client.py)
   - ✅ Server-side packet processing (framework/optimization/_server.py)
   - ✅ Heartbeat/HandShake/Disconnection protocols
   - ✅ Socket-based state management with dict[socket.socket, SocketState]
   - ✅ Non-blocking select() for connection handling
   - ✅ CMA-ES integration with new packet system
   - ✅ **DEPLOYED**: New implementation has replaced the old system

### Key Files
- `framework/types/communication.py`: Communication foundation classes (PacketType, Packet, SocketState)
- `framework/optimization/_client.py`: Multithreaded client implementation with packet-based communication
- `framework/optimization/_server.py`: Packet-based server implementation with socket state management
- `framework/optimization/_distribution.py`: Socket-based distribution system
- `framework/optimization/_connection_utils.py`: Packet send/receive utilities