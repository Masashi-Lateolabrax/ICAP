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
    from scheme.pheromone_exploration.src import optimization, analysis, recode, IncreaseData, DecreaseData
    from libs.utils import get_head_hash

    head_hash = get_head_hash()[0:8]

    hist = optimization()
    hist.save(os.path.join(workdir, f"history_{head_hash}.npz"))

    para = hist.get_min().min_para
    data_inc = IncreaseData(para)
    data_dec = DecreaseData(data_inc)

    recode(para, data_inc, data_dec, workdir)

    analysis(workdir, para, data_inc, data_dec)


if __name__ == '__main__':
    cd = os.path.dirname(os.path.abspath(__file__))
    wd = prepare_workdir(cd)

    generate(os.path.join(cd, "settings.yaml"), os.path.join(cd, "src/settings.py"))

    main(wd)
