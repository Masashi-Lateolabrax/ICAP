import os
from datetime import datetime

from libs.setting import generate


def prepare_workdir(current_dir):
    from libs.utils import get_head_hash

    head_hash = get_head_hash()[0:8]
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    workdir = os.path.join(current_dir, f"results/{current_time_str}_{head_hash}")

    if not os.path.exists(workdir):
        os.makedirs(workdir)

    return workdir


def main(workdir):
    from scheme.pushing_food_with_pheromone.src import optimization, analysis, record

    git_hash = os.path.basename(workdir)[-8:]

    hist = optimization()
    hist.save(os.path.join(workdir, f"history_{git_hash}.npz"))

    para = hist.get_min().min_para

    record(para, workdir)

    analysis(workdir, para)


def test_rec(workdir):
    from libs.optimizer import Hist

    from scheme.pushing_food_with_pheromone.src import record

    git_hash = os.path.basename(workdir)[-8:]

    hist = Hist.load(os.path.join(workdir, f"history_{git_hash}.npz"))
    para = hist.get_min().min_para

    record(para, workdir)


def analysis_only(work_dir):
    from libs.optimizer import Hist

    import scheme.pushing_food_with_pheromone.src as src

    git_hash = os.path.basename(work_dir)[-8:]

    hist = Hist.load(os.path.join(work_dir, f"history_{git_hash}.npz"))
    para = hist.get_min().min_para

    src.analysis(work_dir, para)


def test_xml():
    from scheme.pushing_food_with_pheromone.src.world import gen_xml
    print(gen_xml())


if __name__ == '__main__':
    cd = os.path.dirname(os.path.abspath(__file__))
    # wd = prepare_workdir(cd)
    wd = os.path.join(cd, f"results/20240831_045005_df7c77fa")

    generate(os.path.join(cd, "settings.yaml"), os.path.join(cd, "src/settings.py"))

    # main(wd)
    # test_xml()
    # test_rec(wd)
    analysis_only(wd)
