import warnings
from datetime import datetime
import shutil

import os

from settings import Settings, EType


def prepare_dir(current_dir, specify: str = None):
    from libs.utils import get_head_hash

    head_hash = get_head_hash()[0:8]

    if specify is None:
        current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        workdir = os.path.join(current_dir, f"results/{current_time_str}_{head_hash}")
    else:
        workdir = os.path.join(current_dir, f"results/{specify}")
        if head_hash != specify[-8:]:
            warnings.warn("Your specified git hash don't match the HEAD's git hash.")

    if not os.path.exists(workdir):
        os.makedirs(workdir)
        shutil.copy(os.path.join(current_dir, "Note.md"), workdir)

    return workdir


def main(workdir):
    import scheme.pushing_food_with_pheromone.src as src

    best_para = src.optimization(workdir)
    src.sampling(workdir, best_para)


def sampling(workdir):
    import scheme.pushing_food_with_pheromone.src as src

    gen_dir = os.path.join(workdir, "best")
    os.makedirs(gen_dir, exist_ok=True)

    loader = src.LogLoader(workdir)
    if Settings().Optimization.EVALUATION_TYPE == EType.POTENTIAL:
        para = loader.get_max_individual()
    else:
        para = loader.get_min_individual()

    src.sampling(gen_dir, para)


def plot_evaluations(workdir):
    import scheme.pushing_food_with_pheromone.src as src
    loader = src.LogLoader(workdir)

    src.plot_evaluation_for_each_generation(workdir, loader)
    src.plot_evaluation_elements_for_each_generation(workdir, loader)


if __name__ == '__main__':
    cd = os.path.dirname(os.path.abspath(__file__))
    wd = prepare_dir(cd)

    main(wd)
    # sampling(wd)
    # plot_evaluations(wd)
