#!/bin/bash

# Get number of CPU cores
cpu_cores=$(nproc)

echo "Detected $cpu_cores CPU cores"
echo "Starting $cpu_cores main.py processes..."

# Start processes in background
for i in $(seq 1 $cpu_cores); do
    echo "Starting process $i..."
    PYTHONPATH=. uv run src/main.py &
done

# Wait for all background processes to complete
wait

echo "All processes completed."