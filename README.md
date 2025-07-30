# Self-Organized Robotic Behavior with Pheromones

Experimental code for training robots to learn self-organized behavior using pheromone mechanisms in simulation environments.

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

## Environment Setup

This project uses [uv](https://github.com/astral-sh/uv) as the Python package manager. Run the following commands to set
up the environment for running experiments:

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

> **Troubleshooting**: If you encounter PIL-related errors, see the [Pillow Compatibility Note](#pillow-compatibility-note) at the end of this document.

## Usage

This project implements a distributed optimization system using CMA-ES (Covariance Matrix Adaptation Evolution Strategy) with a client-server architecture for training robotic behaviors.

### Running the Optimization System

**Important**: All commands must be executed from the project root directory (`/path/to/ICAP/`).

#### 1. Start the Optimization Server

The server manages the CMA-ES optimization process and coordinates multiple client connections:

```bash
# Navigate to project root directory first
cd /path/to/ICAP

# For CUDA 12.8 support
PYTHONPATH=. uv run --extra cu128 src/server.py

# For CUDA 12.4 support  
PYTHONPATH=. uv run --extra cu124 src/server.py

# For CPU-only execution
PYTHONPATH=. uv run --extra cpu src/server.py
```

#### 2. Connect Optimization Clients

Clients perform the actual robot simulation evaluations using multiprocessing for parallel evaluation.

```bash
# Navigate to project root directory first
cd /path/to/ICAP

# Single evaluation process (default)
PYTHONPATH=. uv run --extra cu128 src/client.py

# Multiple evaluation processes for parallel simulation
PYTHONPATH=. uv run --extra cu128 src/client.py --num-processes 4

# With custom server settings
PYTHONPATH=. uv run --extra cu128 src/client.py --host 192.168.1.100 --port 5001 --num-processes 4
```

Replace `cu128` with `cu124` or `cpu` depending on your environment setup.


## Project Structure

- **`src/`**: Main application entry points (client.py, server.py, analysis.py)
- **`framework/`**: Core framework implementation
  - `optimization/`: CMA-ES server and client with packet-based communication
  - `backends/`: Physics simulation backends (MuJoCo, Genesis-World)
  - `sensor/`: Robot sensor implementations (omni-sensor, direction sensor)
  - `types/`: Data structures and type definitions
  - `environment/`: Environment setup and object positioning
- **`examples/`**: Example implementations and test scripts
- **`assets/`**: 3D models for robots and food objects
- **`results/`**: Experiment results organized by timestamp


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
|------------|---------------------|-----------|--------------------|
| `main`     | Production code     | -         | -                  |
| `develop`  | Shared development  | `main`    | `main`, `scheme/*` |
| `scheme/*` | Experiment-specific | `develop` | `main`             |

## Pillow Compatibility Note

By default, `uv` uses a standalone Python environment, which may cause errors when importing `PIL._imagingtk`:

```
TypeError: bad argument type for built-in operation
```

This happens because the `_imagingtk` C extension isn't properly built.

This issue is known to occur with uv when it auto-installs python-build-standalone, which lacks full support for certain
C extensions. See also:
[astral-sh/python-build-standalone#533](https://github.com/astral-sh/python-build-standalone/issues/533)

### Solution

1. Install dependencies:

   ```bash
   sudo apt install tk-dev tcl-dev
   ```

2. Install Python via `pyenv`:

   ```bash
   pyenv install 3.12.10
   ```

3. Remove uv's standalone Python if not needed:

   ```bash
   rm -rf ~/.local/share/uv/python
   ```

This ensures full Pillow functionality, including `ImageTk`.

## References

To be added as research progresses.