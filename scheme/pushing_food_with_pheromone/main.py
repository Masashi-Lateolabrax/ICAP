import os

from libs.setting import generate


def prepare_workdir(current_dir):
    from datetime import datetime
    from libs.utils import get_head_hash

    head_hash = get_head_hash()[0:8]
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    workdir = os.path.join(current_dir, f"results/{current_time_str}_{head_hash}")

    if not os.path.exists(workdir):
        os.makedirs(workdir)

    return workdir


def main(workdir):
    from scheme.pushing_food_with_pheromone.src import optimization, analysis, record
    from libs.utils import get_head_hash

    head_hash = get_head_hash()[0:8]

    hist = optimization()
    hist.save(os.path.join(workdir, f"history_{head_hash}.npz"))

    para = hist.get_min().min_para

    record(para, workdir)

    analysis()


def test_rec(workdir):
    from libs.optimizer import Hist
    from libs.utils import get_head_hash

    from scheme.pushing_food_with_pheromone.src import record

    head_hash = get_head_hash()[0:8]

    hist = Hist.load(os.path.join(workdir, f"history_{head_hash}.npz"))
    para = hist.get_min().min_para

    record(para, workdir)


def test_xml():
    from scheme.pushing_food_with_pheromone.src.world import gen_xml
    print(gen_xml())


if __name__ == '__main__':
    cd = os.path.dirname(os.path.abspath(__file__))
    wd = prepare_workdir(cd)
    # wd = os.path.join(cd, "results/20240830_173155_70cb925e")

    generate(os.path.join(cd, "settings.yaml"), os.path.join(cd, "src/settings.py"))

    main(wd)
    # test_xml()
    # test_rec(wd)
