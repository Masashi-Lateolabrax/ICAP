import warnings
from datetime import datetime
import shutil

import numpy as np

import os
import copy
import pickle
import concurrent.futures
from multiprocessing import Manager


class LogFragment:
    def __init__(self, para):
        self.gen = -1
        self.para = copy.deepcopy(para)
        self.data = []

    def add_score(self, data):
        self.data.append(copy.deepcopy(data))


class _Logger:
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def __init__(self):
        self.fragments = Manager().list()

    def __del__(self):
        if self.executor:
            self.executor.shutdown(wait=True)

    def add_fragment(self, fragment: LogFragment):
        self.fragments.append(fragment)

    def save(self, dir_path, gen):
        fragments = list(self.fragments)

        os.makedirs(dir_path, exist_ok=True)
        for f in fragments:
            f.gen = gen

        log_file: str = os.path.join(dir_path, f"{gen}.pkl")

        try:
            with open(log_file, "bw") as f:
                pickle.dump(fragments, f)
            print(f"Generation {gen} saved successfully.")
        except Exception as e:
            print(f"Failed to save generation {gen}: {e}")
        finally:
            self.fragments[:] = []


def prepare_workdir(current_dir, specify: str = None):
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
        shutil.copy(os.path.join(current_dir, "Note.md"), workdir)

    return workdir


def main(workdir, logger):
    import scheme.pushing_food_with_pheromone.src as src

    git_hash = os.path.basename(workdir)[-8:]

    hist = src.optimization(logger)
    hist.save(os.path.join(workdir, f"history_{git_hash}.npz"))

    para = hist.get_max().max_para
    src.sampling(workdir, para)
    src.plot_loss(workdir, hist)


def rec_only(workdir):
    from libs.optimizer import Hist

    import scheme.pushing_food_with_pheromone.src as src

    git_hash = os.path.basename(workdir)[-8:]

    hist = Hist.load(os.path.join(workdir, f"history_{git_hash}.npz"))

    # for i in [len(hist.queues) - 1]:
    #     gen_dir = os.path.join(workdir, str(i))
    #     os.makedirs(gen_dir, exist_ok=True)
    #     para = hist.queues[i].min_para
    #     src.record(para, gen_dir)

    for name in ["test"]:
        gen_dir = os.path.join(workdir, name)
        os.makedirs(gen_dir, exist_ok=True)
        para = hist.get_max().max_para
        src.record(para, gen_dir)


def analysis_only(work_dir):
    from libs.optimizer import Hist

    import scheme.pushing_food_with_pheromone.src as src

    git_hash = os.path.basename(work_dir)[-8:]

    hist = Hist.load(os.path.join(work_dir, f"history_{git_hash}.npz"))

    # for i in [len(hist.queues) - 1]:
    #     gen_dir = os.path.join(work_dir, str(i))
    #     os.makedirs(gen_dir, exist_ok=True)
    #     src.analysis2(gen_dir, hist)

    for name in ["test"]:
        gen_dir = os.path.join(work_dir, name)
        os.makedirs(gen_dir, exist_ok=True)
        src.analysis2(gen_dir, hist)


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


class Parsed:
    def __init__(self, i, fragment: LogFragment):
        self.i = i

        self.gen = fragment.gen
        self.para = fragment.para
        self.scores = np.array(fragment.data)

        self.latest_evaluation = self.scores[:, 0]
        self.latest_evaluation_fr = self.latest_evaluation[:, 0]
        self.latest_evaluation_fn = self.latest_evaluation[:, 1]

        self.old_evaluation = self.scores[:, 1]
        self.old_evaluation_fr = self.old_evaluation[:, 0]
        self.old_evaluation_fn = self.old_evaluation[:, 1]

        self.latest_sum_evaluation = np.sum(self.latest_evaluation)
        self.old_sum_evaluation = np.sum(self.old_evaluation)


def load_log():
    import pickle
    import scheme.pushing_food_with_pheromone.src as src

    for i in [1, 100, 200, 300, 400]:
        with open(f"./results/logs/{i}.pkl", "br") as f:
            log_list: list[LogFragment] = pickle.load(f)

        log_list: list[Parsed] = [Parsed(j, lf) for j, lf in enumerate(log_list)]
        a: list[Parsed] = list(reversed(sorted(log_list, key=lambda x: x.latest_sum_evaluation)))
        b: list[Parsed] = sorted(log_list, key=lambda x: x.old_sum_evaluation)

        os.makedirs(f"./results/logs/{i}_video/latest", exist_ok=True)
        os.makedirs(f"./results/logs/{i}_video/old", exist_ok=True)

        src.sampling(f"./results/logs/{i}_video/latest", a[0].para)
        src.sampling(f"./results/logs/{i}_video/old", b[0].para)

        with open(f"./results/logs/{i}_video/score.txt", "w") as f:
            f.write(f"latest: {a[0].latest_sum_evaluation}\n")
            f.write(f"latest_idx: {a[0].i}\n")
            f.write(f"old: {b[0].old_sum_evaluation}\n")
            f.write(f"old_idx: {b[0].i}\n")


def plot_ele_eva():
    import pickle
    import scheme.pushing_food_with_pheromone.src as src

    latest = []
    old = []

    for i in range(1, 501):
        with open(f"./results/logs/{i}.pkl", "br") as f:
            log_list: list[LogFragment] = pickle.load(f)
        log_list: list[Parsed] = [Parsed(j, lf) for j, lf in enumerate(log_list)]
        a: list[Parsed] = list(reversed(sorted(log_list, key=lambda x: x.latest_sum_evaluation)))
        b: list[Parsed] = sorted(log_list, key=lambda x: x.old_sum_evaluation)
        latest.append((np.sum(a[0].latest_evaluation_fn), np.sum(a[0].latest_evaluation_fr)))
        old.append((np.sum(b[0].old_evaluation_fn), np.sum(b[0].old_evaluation_fr)))
        print("Processed:", i)

    latest = np.array(latest)
    old = np.array(old)

    src.plot_element_evaluation(f"./results/logs/", latest, old)


if __name__ == '__main__':
    Logger = _Logger()

    cd = os.path.dirname(os.path.abspath(__file__))
    wd = prepare_dir(cd)

    main(wd, Logger)
    # test_xml()
    # rec_only(wd)
    # analysis_only(wd)
    # sampling(wd)
    # load_log()
    # plot_ele_eva()
