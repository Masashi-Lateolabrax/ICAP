#!/bin/bash

# Default values
host=""
port=""
extra="cpu"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            host="$2"
            shift 2
            ;;
        --port)
            port="$2"
            shift 2
            ;;
        --extra)
            extra="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--host HOST] [--port PORT] [--extra EXTRA]"
            echo "  EXTRA options: cu128, cu124, cpu (default: cpu)"
            exit 1
            ;;
    esac
done

# Build command arguments
cmd_args=""
if [[ -n "$host" ]]; then
    cmd_args="$cmd_args --host $host"
fi
if [[ -n "$port" ]]; then
    cmd_args="$cmd_args --port $port"
fi

# Get number of CPU cores
cpu_cores=$(nproc)
process_count=$((cpu_cores * 3 / 4))

echo "Detected $cpu_cores CPU cores"
echo "Starting $process_count main.py processes (3/4 of cores)..."
echo "Using extra: $extra"
if [[ -n "$cmd_args" ]]; then
    echo "Using arguments:$cmd_args"
fi

# Start processes in background
for i in $(seq 1 $process_count); do
    echo "Starting process $i..."
    PYTHONPATH=. uv run --extra $extra src/main.py $cmd_args &
done

# Wait for all background processes to complete
wait

echo "All processes completed."