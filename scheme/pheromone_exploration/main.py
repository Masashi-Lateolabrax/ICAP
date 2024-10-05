import os
from datetime import datetime
import shutil
import warnings

from settings import Settings


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
    import scheme.pheromone_exploration.src as src

    for i in range(Settings.NUM_GENERATION):
        case_dir = os.path.join(workdir, f"case{i}")
        os.makedirs(case_dir, exist_ok=True)
        para = src.gen_parameters()
        data_inc, data_dec = src.dump(case_dir, para)
        src.record(data_inc, data_dec, case_dir)
        src.analysis2(case_dir, data_inc, data_dec)


def rec_only(workdir):
    import scheme.pheromone_exploration.src as src

    para = {
        "sv": Settings.Pheromone.SATURATION_VAPOR,
        "evaporate": Settings.Pheromone.EVAPORATION,
        "diffusion": Settings.Pheromone.DIFFUSION,
        "decrease": Settings.Pheromone.DECREASE
    }
    data_inc, data_dec = src.dump(workdir, para)
    src.record(data_inc, data_dec, workdir)


if __name__ == '__main__':
    cd = os.path.dirname(os.path.abspath(__file__))
    wd = prepare_dir(cd)

    main(wd)
