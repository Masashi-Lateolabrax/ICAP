import subprocess
import gc
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


def check_gpu_temperature():
    """Check GPU temperature"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5, check=True
        )
        return int(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return "N/A"
    except Exception:
        return "N/A"


def check_gpu_utilization():
    """Check GPU utilization percentage"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5, check=True
        )
        return int(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return "N/A"
    except Exception:
        return "N/A"


def force_garbage_collection():
    """Force garbage collection to prevent memory fragmentation"""
    gc.collect()


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


def monitor_gpu_health():
    """Monitor comprehensive GPU health metrics"""
    temp = check_gpu_temperature()
    util = check_gpu_utilization()

    # Check for thermal throttling
    thermal_warning = ""
    if isinstance(temp, int) and temp > 80:
        thermal_warning = " ⚠️ HIGH TEMP"
    elif isinstance(temp, int) and temp > 85:
        thermal_warning = " 🔥 THERMAL THROTTLING"

    # Check for low utilization
    util_warning = ""
    if isinstance(util, int) and util < 50:
        util_warning = " ⚠️ LOW UTILIZATION"

    print(f"GPU Health: {temp}°C, {util}% util{thermal_warning}{util_warning}")
    return {"temperature": temp, "utilization": util}

def monitor_gpu_clocks():
    """Monitor GPU clock frequencies (graphics and memory clocks)"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=clocks.gr,clocks.mem,clocks.max.gr,clocks.max.mem', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                current_gr, current_mem, max_gr, max_mem = line.split(', ')
                gr_percent = (int(current_gr) / int(max_gr)) * 100 if max_gr != '0' else 0
                mem_percent = (int(current_mem) / int(max_mem)) * 100 if max_mem != '0' else 0
                print(f"GPU {i}: Graphics {current_gr}MHz ({gr_percent:.1f}% of {max_gr}MHz), Memory {current_mem}MHz ({mem_percent:.1f}% of {max_mem}MHz)")
            return {"graphics_clock": int(current_gr), "memory_clock": int(current_mem), 
                    "max_graphics_clock": int(max_gr), "max_memory_clock": int(max_mem)}
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        print("GPU clock monitoring unavailable (nvidia-smi not found or failed)")
        return None
    except Exception as e:
        print(f"GPU clock monitoring error: {e}")
        return None


def monitor_comprehensive_gpu():
    """Monitor all GPU metrics: health, memory, and clocks"""
    print("=== GPU Status ===")
    health = monitor_gpu_health()
    monitor_gpu_memory()
    clocks = monitor_gpu_clocks()
    print("==================")
    return {"health": health, "clocks": clocks}
