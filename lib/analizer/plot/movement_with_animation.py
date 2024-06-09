from lib.optimizer import Hist


def plot_parameter_movements_animation_graph(queues: list[Hist.Queue], y_max: float = 1.0):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    x = np.arange(0, queues[0].min_para.shape[0], dtype=np.uint64)

    def frame():
        graphs = [None, None, None]
        length = len(queues) - 1
        for i in range(0, length):
            print(f"{i}/{length}")
            graphs.append(
                np.abs(queues[i + 1].min_para - queues[i].min_para)
            )
            graphs.pop(0)
            yield i, graphs[2], graphs[1], graphs[0]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    def update(f):
        i, first, second, third = f
        std = np.std(first)

        ax.clear()
        ax.set_title(f"Frame {i}")
        ax.set_xlabel("parameters")
        ax.set_ylabel("movements")
        ax.set_ylim(0, y_max)

        artists = [
            ax.plot(x, [std * 3] * len(x), c="Red"),
            ax.plot(x, third, c="Blue", alpha=0.1) if third is not None else [],
            ax.plot(x, second, c="Blue", alpha=0.3) if second is not None else [],
            ax.plot(x, first, c="Blue", alpha=1.0),
        ]

        return sum(artists, [])

    ani = animation.FuncAnimation(fig, update, frames=frame, blit=False, cache_frame_data=False)
    ani.save('animation.mp4')


def main():
    import os

    working_directory = "../../../src/"

    # history = get_current_history(working_directory)
    history = Hist.load(
        os.path.join(working_directory, "history_8a1d803b.npz")
    )

    queues = history.queues[0:-1]
    plot_parameter_movements_animation_graph(queues, 1.8)


if __name__ == '__main__':
    main()
