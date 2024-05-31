def get_current_history(directory_path):
    import os
    from ._git import get_head_hash

    head_hash = get_head_hash()[0:8]
    result = os.path.join(directory_path, f'history_{head_hash}.npz')
    if not os.path.exists(result):
        print(os.path.abspath(result))
        raise f"Failed to find the history_XXXXXXXX.npz. Make sure to check out the correct commit."
    return result


def load_parameter(
        dim: int,
        working_directory: str,
        git_hash: str | None = None,
        queue_index: int = -1
):
    if git_hash is None:
        import numpy as np
        para = np.random.random(dim)
    else:
        from ..optimizer import Hist
        import os
        path = os.path.join(working_directory, f"history_{git_hash[0:8]}.npz")
        history = Hist.load(path)
        para = history.queues[queue_index].min_para

    return para
