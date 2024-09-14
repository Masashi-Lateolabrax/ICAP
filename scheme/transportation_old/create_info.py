import os

from lib.optimizer import Hist


def main():
    cd = os.path.dirname(__file__)

    for i in range(0, 22):
        task_dir = os.path.join(cd, f"results/sample{i}")
        if not os.path.exists(task_dir):
            print(f"Skip `sample{i}`")
            continue

        hist_path = os.path.join(task_dir, "history_77102066.npz")
        hist = Hist.load(hist_path)

        info_path = os.path.join(task_dir, "info.txt")
        with open(info_path, "w") as f:
            f.write(f"generation: {len(hist.queues)}\n")
            f.write(f"mu: {hist.mu}\n")
            f.write(f"dim: {hist.dim}\n")
            f.write(f"population: {hist.population}\n")


if __name__ == '__main__':
    main()
