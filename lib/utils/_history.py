def get_current_history(directory_path):
    import os
    from ._git import get_head_hash

    head_hash = get_head_hash()[0:8]
    result = os.path.join(directory_path, f'history_{head_hash}.npz')
    if not os.path.exists(result):
        print(os.path.abspath(result))
        raise f"Failed to find the history_XXXXXXXX.npz. Make sure to check out the correct commit."
    return result


def load_parameter(dim: int, load_history_file: str | None = "", queue_index: int = -1):
    import numpy as np
    from ..optimizer import Hist

    if isinstance(load_history_file, str):
        if load_history_file == "":
            history = Hist.load(get_current_history("../../../"))
        else:
            history = Hist.load(load_history_file)
        para = history.queues[queue_index].min_para
    else:
        para = np.random.random(dim)

    return para
