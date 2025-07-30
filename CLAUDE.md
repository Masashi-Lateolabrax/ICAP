# ICAP Project Notes

## Project Overview
**ICAP (Intelligent Collective Autonomous Phenomenology)** is an experimental robotics research platform that studies self-organized robotic behavior using pheromone-based mechanisms. The project implements a distributed optimization system that trains robots to learn collective behaviors through swarm intelligence, specifically focusing on food foraging and transport tasks.

### Research Goals
- Train robots to develop self-organized behaviors through collective intelligence
- Evaluate pheromone mechanisms' contribution to robotic swarm behavior  
- Analyze relationships between environmental parameters and emergent behaviors
- Create an explainable model for understanding collective robotics

## Architecture & Performance
- **Server**: Single-threaded CMA-ES optimization server with packet-based communication
- **Clients**: Multiple processes for distributed fitness evaluation with multithreaded architecture
- **Communication**: Packet-based socket system with structured protocol (HANDSHAKE, HEARTBEAT, REQUEST, RESPONSE, DISCONNECTION, ACK)
- **CPU Usage**: Process-based parallelism with threading for client communication
- **SOCKET_BACKLOG**: Controls TCP queue size, not parallel processing
- **Physics**: MuJoCo integration for realistic robot dynamics
- **Dependencies**: Genesis-World framework integration alongside MuJoCo

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
- `optimization/_server.py`: Core server implementation with packet-based communication
- `optimization/_client.py`: Multithreaded client with communication handling
- `backends/_mujoco.py`: MuJoCo physics backend
- `backends/init_genesis.py`: Genesis-World integration
- `core/simulator.py`: Core simulation engine (empty - being refactored)
- `core/train.py`: Training logic (empty - being refactored)
- `sensor/`: Robot sensor implementations (omni-sensor, direction sensor)
- `config/_settings.py`: Base framework settings
- `types/`: Data structures and type definitions including communication protocols
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
- **Physics**: `mujoco==3.3.3`, `genesis-world` (from Genesis-Embodied-AI)
- **Optimization**: `cmaes==0.11.1`, `scipy==1.16.0`
- **Neural Networks**: `torch` (CPU/CUDA 12.4/12.8 support)
- **Utilities**: `icecream==2.1.5` (debugging), `pillow==11.2.1`, `glfw==2.9.0`
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
- **Type Consistency Errors**: Always verify function signatures match usage patterns - check both parameter types and return types before implementation
- **Array Shape Assumptions**: Never assume array shapes without explicit validation - always check sensor outputs and data structure consistency

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

## Server Performance Optimization Analysis (2025-07-08)

### Background Issue
During execution of `src/main_parallel.py` with `collect_throughput_observations`, socket timeout warnings appear: "WARNING:root:Socket timeout on attempt 1 while receiving data". This occurs because the adaptive client manager rapidly creates and destroys client processes to measure throughput at different process counts, overwhelming the server's socket handling capacity.

### Root Cause Analysis
The timeout warnings stem from configuration mismatches and performance bottlenecks:

1. **Timeout Configuration Conflicts**:
   - Client socket timeout: 10 seconds (`framework/optimization/_client.py:32`)
   - Server dead socket detection: 30 seconds (`framework/optimization/_server.py:21`)
   - Heartbeat interval: 20 seconds (`framework/optimization/_client.py:15`)
   - Server select timeout: 20 seconds (`framework/optimization/_server.py:116`)

2. **Performance Bottlenecks**:
   - `collect_throughput_observations()` in `framework/optimization/_parallel.py:156-180`
   - Rapid client process creation/destruction cycle
   - Single-threaded server processing with blocking CMA-ES updates
   - Inefficient socket polling and connection management

### Current Server Architecture Issues

#### Connection Management (`framework/optimization/_server.py:116-189`)
- **Problem**: `select()` with 20-second timeout blocks new connections
- **Impact**: New clients timeout before server processes their handshake
- **Location**: `_communicate_with_client()` method

#### CMA-ES Integration (`framework/optimization/_server.py:200-228`)
- **Problem**: `cmaes.update()` blocks main loop during optimization
- **Impact**: No new connections processed during CMA-ES calculations
- **Location**: Main server loop in `run()` method

#### Socket Health Monitoring (`framework/optimization/_server.py:99-104`)
- **Problem**: 30-second timeout for dead socket detection too slow
- **Impact**: Server continues attempting communication with disconnected clients
- **Location**: `_mut_drop_dead_sockets()` method

### Proposed Server Response Speed Optimizations

#### 1. Communication Timeout Adjustments (High Priority)
```python
# Current problematic settings
SOCKET_TIMEOUT = 30              # Too long - reduce to 15
server_socket.settimeout(1.0)    # OK for accept()
select(timeout=20.0)             # Too long - reduce to 1.0

# Optimized settings
SOCKET_TIMEOUT = 15              # Faster dead connection detection
select(timeout=1.0)              # More responsive polling
```

#### 2. Non-blocking Socket Operations (High Priority)
- **Current**: Sequential socket processing in `_communicate_with_client()`
- **Optimization**: Batch process multiple readable sockets
- **Benefit**: Reduced latency for multiple concurrent connections

#### 3. Asynchronous CMA-ES Updates (High Impact)
- **Current**: CMA-ES blocks main server loop
- **Optimization**: Move CMA-ES to separate thread
- **Benefit**: Continuous client connection processing during optimization

#### 4. Connection Queue Optimization (Medium Priority)
- **Current**: Linear queue checking in `_mut_retrieve_socket()`
- **Optimization**: Event-driven connection acceptance
- **Benefit**: Immediate processing of new connections

#### 5. Individual Assignment Optimization (Medium Priority)
- **Current**: All sockets checked every iteration
- **Optimization**: Track work-ready sockets separately
- **Benefit**: Reduced CPU overhead for client assignment

#### 6. Debug Output Reduction (Easy Implementation)
- **Current**: Extensive `ic()` debug output throughout server
- **Optimization**: Conditional debug output or async logging
- **Benefit**: Reduced I/O blocking during communication

### Implementation Priority
1. **Immediate (High Impact, Low Risk)**:
   - Reduce `select()` timeout from 20s to 1s
   - Reduce `SOCKET_TIMEOUT` from 30s to 15s
   - Disable/reduce debug output in production

2. **Short-term (High Impact, Medium Risk)**:
   - Implement non-blocking socket batch processing
   - Add connection pool management
   - Optimize individual assignment tracking

3. **Long-term (High Impact, High Risk)**:
   - Asynchronous CMA-ES updates in separate thread
   - Event-driven connection management
   - Comprehensive performance monitoring

### Expected Outcomes
These optimizations should significantly improve server responsiveness during `collect_throughput_observations` execution, reducing socket timeout warnings and enabling more stable adaptive client management for performance measurement.

### Related Files
- `framework/optimization/_server.py`: Main server implementation
- `framework/optimization/_parallel.py`: Adaptive client manager
- `framework/optimization/_connection_utils.py`: Socket communication utilities
- `framework/config/_settings.py`: Timeout configuration
- `src/settings.py`: Application-specific timeout overrides

## Communication Protocol Bug Fix (2025-07-09)

### Problem Identified
**Root Cause**: Server was not sending ACK response to REQUEST packets, causing communication protocol breakdown.

### Symptoms
1. **Server Side**: `WARNING:root:Received empty chunk, connection may be broken`
2. **Client Side**: `ERROR:root:Received invalid ACK packet`

### Analysis Process
1. **Initial Hypothesis**: Server's `while True` loop causing premature connection drops
2. **Key Insight**: Client maintains connection until ACK received, so empty chunks shouldn't occur
3. **Critical Discovery**: REQUEST packets weren't receiving proper ACK responses
4. **Protocol Violation**: `_deal_with_request()` sent RESPONSE packets instead of ACK packets

### The Fix (Commit 6c6a3fb)
**Before** (Incorrect):
```python
def _deal_with_request(self, request_packets: dict[socket.socket, Packet]):
    for sock, packet in request_packets.items():
        # ... validation code ...
        response_packet = Packet(PacketType.RESPONSE, data=self.socket_states[sock].assigned_individuals)
        if send_packet(sock, response_packet, retry=3) != CommunicationResult.SUCCESS:
            logging.error(f"Failed to send RESPONSE packet to {self.sock_name(sock)}")
            self._drop_socket(sock)
```

**After** (Correct):
```python
def _deal_with_request(self, request_packets: dict[socket.socket, Packet]):
    for sock, packet in request_packets.items():
        # ... validation code ...
        self._response_ack(sock, data=self.socket_states[sock].assigned_individuals)
```

### Technical Details
- **Modified `_response_ack()` method**: Added optional `data` parameter to send ACK with payload
- **Unified Protocol**: All packet types now receive ACK responses through consistent interface
- **Fixed Communication Flow**: REQUEST → ACK (with Individual data) → proper client continuation

### Debugging Lessons Learned
1. **Communication Protocol Analysis**: Always trace complete request-response cycles
2. **Log Correlation**: Match client and server error messages to identify protocol mismatches
3. **Assumption Validation**: Question initial hypotheses when evidence contradicts expectations
4. **Code Review**: Verify all packet types follow consistent ACK response patterns

### Prevention Guidelines
1. **Packet Handler Consistency**: All `_deal_with_*()` methods must send appropriate ACK responses
2. **Protocol Documentation**: Document expected response type for each packet type
3. **Unit Testing**: Test communication protocol completeness for all packet types
4. **Error Message Clarity**: Include packet type information in communication error logs

## Current Project Status (2025-07-09)

### Project State Summary
- **Active Branch**: `develop` (latest commit: 43dc4b8 - "feat: Add main.sh to manage parallel execution of main.py processes")
- **Branch Status**: 2 commits ahead of origin/develop
- **System Status**: ✅ **All bugs fixed and system stable**
- **Communication System**: ✅ **Fully deployed packet-based architecture**
- **Physics Integration**: ✅ **Dual backend support (MuJoCo + Genesis-World)**

### Recent Achievements (Latest Commits)
1. **Parallel Execution Management** (commit 43dc4b8):
   - Added `src/main.sh` for managing parallel client processes
   - Improved system scalability for distributed optimization

2. **Performance Optimizations** (commit 86f1327):
   - Disabled `ic()` debug output for improved performance
   - Reduced I/O overhead in production environment

3. **Task Management Enhancement** (commit 834a1d5):
   - Optimized task removal in `set_evaluated_task` method
   - Improved memory management in optimization server

4. **Bug Fixes & Stability** (commits 7354a5b, f2dd0e4):
   - Fixed server and client communication issues
   - Resolved protocol handling bugs identified in previous analysis

### System Architecture Status
- **Communication Protocol**: ✅ **Fully functional packet-based system**
  - All 6 packet types (HANDSHAKE, HEARTBEAT, REQUEST, RESPONSE, DISCONNECTION, ACK) working correctly
  - Socket timeout warnings resolved
  - Connection management stable
  
- **Distributed Optimization**: ✅ **Production ready**
  - CMA-ES server with stable client coordination
  - Multithreaded client architecture functioning properly
  - Parallel process management via `main.sh`

- **Physics Backends**: ✅ **Dual support implemented**
  - MuJoCo integration: Complete and tested
  - Genesis-World integration: Available and functional
  - Backend selection via configuration

### Current Configuration (Production)
- **Socket Timeout**: Optimized for stability
- **Heartbeat Interval**: 20 seconds
- **Request Limit**: 1 per client
- **Python Version**: 3.12.10 (locked)
- **Population Size**: 1000 (CMA-ES)
- **Simulation Time**: 60s per evaluation

### Branch Strategy & Development
- **Main Branch**: `main` (production releases)
- **Development Branch**: `develop` (current: 43dc4b8)
- **Experimental Branch**: `exp` (feature testing)
- **Scheme Branches**: Multiple experiment-specific branches available

### Key Accomplishments
1. **✅ Communication System**: Packet-based architecture fully deployed and stable
2. **✅ Bug Resolution**: All socket timeout and protocol issues resolved
3. **✅ Performance**: Optimized for production with debug overhead removed
4. **✅ Scalability**: Parallel execution management implemented
5. **✅ Dual Physics**: Both MuJoCo and Genesis-World backends functional

### System Readiness
- **Research Platform**: ✅ Ready for experiments
- **Distributed Computing**: ✅ Stable multi-client architecture
- **Physics Simulation**: ✅ Realistic robot dynamics via MuJoCo/Genesis
- **Optimization**: ✅ CMA-ES with 1000 population size functional
- **Documentation**: ✅ Comprehensive project documentation maintained

### Notable Technical Achievements
1. **Protocol Fix**: Resolved REQUEST packet ACK response issue
2. **Performance**: Eliminated debug output overhead
3. **Scalability**: Added parallel process management
4. **Stability**: Comprehensive error handling and connection management
5. **Flexibility**: Dual physics backend support for different research needs

## Project Management Updates

### Project Status Overview
- **プロジェクト全体を見て，現状をまとめてください．ちなみにバグは全部修正されました．**
  - 全ての既知のバグを修正
  - 通信プロトコルの完全な再実装と最適化
  - マルチスレッドクライアントアーキテクチャの安定化
  - Genesis-Worldとの物理エンジン統合
  - 分散最適化システムの堅牢性向上