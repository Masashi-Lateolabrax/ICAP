import numpy as np


class ActivationInvestigator:
    def __init__(self):
        self.input_buffer = np.zeros(0)
        self.output_buffer = np.zeros(0)

    def update(self, i: np.ndarray, o: np.ndarray):
        if self.input_buffer.shape != i.shape:
            self.input_buffer = np.zeros(i.shape)
        np.copyto(self.input_buffer, i)

        if self.output_buffer.shape != o.shape:
            self.output_buffer = np.zeros(o.shape)
        np.copyto(self.output_buffer, o)


def main():
    from lib.utils import load_parameter, get_head_hash
    from src.exp.the_simplest_task.settings import HyperParameters, TaskGenerator

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    working_directory = "../../../../"
    episode = int(HyperParameters.Simulator.EPISODE / HyperParameters.Simulator.TIMESTEP)

    task_generator = TaskGenerator()
    para = load_parameter(
        dim=task_generator.get_dim(),
        working_directory=working_directory,
        git_hash="904e8bb8",
        queue_index=-1,
    )

    def frame():
        task = task_generator.generate(para)
        activation = ActivationInvestigator()

        task.brain.sequence[1].register_forward_hook(
            lambda _m, i, o: activation.update(i[0].detach().numpy(), o.detach().numpy())
        )

        for t in range(episode):
            print(f"{t}/{episode}")
            task.calc_step()
            yield t, activation

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    def update(x):
        i: int = x[0]
        activation: ActivationInvestigator = x[1]

        ax.clear()
        ax.set_title(f"Frame {i}/{episode}")

        data = np.expand_dims(activation.input_buffer, 0)
        ax.matshow(data)

        return fig.artists

    ani = animation.FuncAnimation(fig, update, frames=frame, blit=False, cache_frame_data=False)
    ani.save('animation.mp4')


if __name__ == '__main__':
    main()
