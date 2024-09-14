import os

import matplotlib.pyplot as plt

from lib.optimizer import Hist


def each_loss():
    cd = os.path.dirname(__file__)
    for i in range(0, 22):
        task_dir = os.path.join(cd, f"results/sample{i}")
        if not os.path.exists(task_dir):
            print(f"Skip `sample{i}`")
            continue

        hist_path = os.path.join(task_dir, "history_77102066.npz")
        hist = Hist.load(hist_path)

        score = [q.min_score for q in hist.queues]
        xs = [i for i in range(0, len(score))]

        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        axis.set_ylim(725, 900)
        axis.plot(xs, score)
        fig.savefig(os.path.join(task_dir, "loss.svg"))
        plt.close(fig)


def comprehensive_loss():
    cd = os.path.dirname(__file__)
    data = []
    for i in range(0, 21):
        task_dir = os.path.join(cd, f"results/sample{i}")
        if not os.path.exists(task_dir):
            print(f"Skip `sample{i}`")
            continue

        hist_path = os.path.join(task_dir, "history_77102066.npz")
        hist = Hist.load(hist_path)

        score = [q.min_score for q in hist.queues]
        data.append(score)

    xs = [i for i in range(len(data[0]))]

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1)
    for i, s in enumerate(data):
        print(i)
        axis.plot(xs, s)
    fig.savefig(os.path.join(cd, "results/c_loss.svg"))


def main():
    each_loss()
    comprehensive_loss()


if __name__ == '__main__':
    main()
