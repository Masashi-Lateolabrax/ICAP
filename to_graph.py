from studyLib import optimizer
from matplotlib import pyplot as plt

if __name__ == '__main__':
    def main():
        hist = optimizer.Hist()
        hist.load("./history.log")

        x = [x for x in range(0, len(hist.queues))]

        fig = plt.figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(1, 1, 1)

        averages = [q.average for q in hist.queues]
        ax.plot(x, averages, color="g")

        min_values = [q.min for q in hist.queues]
        ax.plot(x, min_values, color="b")

        # max_values = [q.max for q in hist.queues]
        # ax.plot(x, max_values, color="r")

        fig.savefig("history.png")
        fig.show()


    main()
