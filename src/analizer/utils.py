import numpy as np


def load_parameter(dim: int, load_history_file: str | None = "", queue_index: int = -1):
    from src.optimizer import Hist
    from src.utils import get_current_history

    if isinstance(load_history_file, str):
        if load_history_file == "":
            history = Hist.load(get_current_history("../../"))
        else:
            history = Hist.load(load_history_file)
        para = history.queues[queue_index].min_para
    else:
        para = np.random.random(dim)

    return para
