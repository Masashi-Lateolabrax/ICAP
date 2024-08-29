import os

from libs.setting import generate


def prepare_workdir():
    from datetime import datetime
    from libs.utils import get_head_hash

    current_dir = os.path.dirname(os.path.abspath(__file__))

    head_hash = get_head_hash()[0:8]
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    workdir = os.path.join(current_dir, f"results/{current_time_str}_{head_hash}")

    if not os.path.exists(workdir):
        os.makedirs(workdir)

    return current_dir, workdir


def main(workdir):
    from scheme.simple_transportation.src import optimization, analysis, record
    from libs.utils import get_head_hash

    head_hash = get_head_hash()[0:8]

    hist = optimization()
    hist.save(os.path.join(workdir, f"history_{head_hash}.npz"))

    para = hist.get_min().min_para

    record(para, workdir)

    analysis()


def test_run(workdir):
    import numpy as np
    import scheme.simple_transportation.src as src

    rng = np.random.default_rng()
    para = rng.random(
        src.TaskGenerator().get_dim()
    )

    src.record(para, workdir)


if __name__ == '__main__':
    cd, wd = prepare_workdir()

    generate(os.path.join(cd, "settings.yaml"), os.path.join(cd, "src/settings.py"))

    main(wd)
    # test_run(wd)
