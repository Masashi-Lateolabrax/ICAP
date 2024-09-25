import os
from datetime import datetime
import shutil
import warnings
import json

from settings import Settings


def prepare_workdir(current_dir):
    from datetime import datetime
    from libs.utils import get_head_hash

    head_hash = get_head_hash()[0:8]
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    workdir = os.path.join(current_dir, f"results/{current_time_str}_{head_hash}")

    if not os.path.exists(workdir):
        os.makedirs(workdir)

    return workdir


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

    shutil.copy(os.path.join(current_dir, "settings.py"), workdir)
    shutil.copy(os.path.join(current_dir, "Note.md"), workdir)

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


def analyse_charactor(workdir):
    import scheme.pheromone_exploration.src as src

    para = src.convert_characteristic(
        sv=10,
        evaporation=20,
        decrease=0.1,
        diffusion=35
    )

    data_inc = src.IncreaseData(para)
    data_dec = src.DecreaseData(data_inc)

    src.recode(para, data_inc, data_dec, workdir)
    src.analysis(workdir, para, data_inc, data_dec)


if __name__ == '__main__':
    cd = os.path.dirname(os.path.abspath(__file__))
    wd = prepare_workdir(cd)

    generate(os.path.join(cd, "settings.yaml"), os.path.join(cd, "src/settings.py"))

    main(wd)
