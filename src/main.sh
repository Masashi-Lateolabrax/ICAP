#!/bin/bash

# Get number of CPU cores
cpu_cores=$(nproc)
process_count=$((cpu_cores * 3 / 4))

echo "Detected $cpu_cores CPU cores"
echo "Starting $process_count main.py processes (3/4 of cores)..."

# Start processes in background
for i in $(seq 1 $process_count); do
    echo "Starting process $i..."
    PYTHONPATH=. uv run src/main.py &
done

# Wait for all background processes to complete
wait

echo "All processes completed."