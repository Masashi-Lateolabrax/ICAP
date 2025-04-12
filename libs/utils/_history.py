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
        path = os.path.abspath(
            os.path.join(working_directory, f"history_{git_hash[0:8]}.npz")
        )
        history = Hist.load(path)
        if queue_index is None:
            para = history.get_min().min_para
        else:
            para = history.queues[queue_index].min_para

    return para
