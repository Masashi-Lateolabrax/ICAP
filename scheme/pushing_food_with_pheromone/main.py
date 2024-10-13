import warnings
from datetime import datetime
import shutil

import os


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
    from libs.optimizer import Hist
    import scheme.pushing_food_with_pheromone.src as src

    git_hash = os.path.basename(workdir)[-8:]

    hist = Hist.load(os.path.join(workdir, f"history_{git_hash}.npz"))
    # src.plot_loss(workdir, hist)

    # for i in [len(hist.queues) - 1]:
    #     gen_dir = os.path.join(work_dir, str(i))
    #     os.makedirs(gen_dir, exist_ok=True)
    #     src.analysis2(gen_dir, hist)

    for name, para in [(f"sample{i}", hist.get_max().max_para) for i in range(10)]:
        gen_dir = os.path.join(workdir, name)
        os.makedirs(gen_dir, exist_ok=True)
        src.sampling(gen_dir, para)


if __name__ == '__main__':
    cd = os.path.dirname(os.path.abspath(__file__))
    wd = prepare_dir(cd)

    main(wd)
    # test_xml()
    # rec_only(wd)
    # analysis_only(wd)
    # sampling(wd)
    # load_log()
    # plot_ele_eva()
