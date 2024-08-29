import subprocess


def get_head_hash() -> str | None:
    command = ['git', 'rev-parse', 'HEAD']
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        return None
