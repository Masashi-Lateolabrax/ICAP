import io
import pandas
from studyLib import optimizer
from matplotlib import pyplot as plt

# raw = """
# 100,10310.870979938441
# 200,8474.413272543963
# 300,6659.893735300784
# 400,4882.888537884287
# 500,3140.9600026964404
# 600,1402.835515564032
# 700,230.7140427334744
# 800,114.5039288466175
# 900,76.50067211495646
# 1000,57.165075008471355
# 1100,45.26236209388445
# 1200,37.10043064259376
# 1300,31.07631963489244
# 1400,26.975905680042207
# 1500,23.8568588370372
# 1600,21.345645238554376
# 1700,19.337177324702196
# 1800,17.60265953531747
# 1900,16.13042518642169
# 2000,14.95187767620363
# """

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

        # df = pandas.read_csv(io.StringIO(raw), header=None)
        # fig = plt.figure(figsize=(6, 4), dpi=100)
        # ax = fig.add_subplot(1, 1, 1)
        # ax.plot(df[0], df[1], color="g")
        # fig.savefig("history.png")


    main()
