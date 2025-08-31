import subprocess

import os
import jax


def configure_gpu_optimization():
    """Configure JAX for optimal GPU utilization"""

    # Enable XLA optimizations (compatible flags)
    os.environ['XLA_FLAGS'] = (
        '--xla_gpu_enable_latency_hiding_scheduler=true '
        '--xla_gpu_enable_highest_priority_async_stream=true '
        '--xla_gpu_deterministic_ops=false '
        '--xla_gpu_autotune_level=4'
    )

    # JAX configuration for GPU memory management
    os.environ['JAX_ENABLE_X64'] = 'False'  # Use 32-bit for better GPU performance

    try:
        # First check available platforms
        available_devices = jax.devices()
        gpu_devices = [d for d in available_devices if d.platform == 'gpu']

        if gpu_devices:
            jax.config.update('jax_platform_name', 'gpu')
            print(f"GPU optimization configured. Available GPUs: {len(gpu_devices)}")
            return True
        else:
            print("Warning: No GPU devices found, using CPU with optimizations")
            # Apply CPU optimizations instead
            os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=true'
            return False

    except Exception as e:
        print(f"Platform detection failed: {e}. Using CPU optimizations.")
        os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=true'
        return False


def monitor_gpu_memory():
    """Monitor GPU memory usage"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                memory_used, memory_total, gpu_util = line.split(', ')
                memory_usage_pct = (int(memory_used) / int(memory_total)) * 100
                print(f"GPU {i}: {gpu_util}% util, {memory_usage_pct:.1f}% memory ({memory_used}MB/{memory_total}MB)")
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        print("GPU monitoring unavailable (nvidia-smi not found or failed)")
    except Exception as e:
        print(f"GPU monitoring error: {e}")
