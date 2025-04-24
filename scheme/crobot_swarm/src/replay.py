import dataclasses
import os.path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter

import framework

from brain import BrainBuilder
from settings import Settings
from logger import Logger


@dataclasses.dataclass
class RobotInfo:
    position: np.ndarray
    input: np.ndarray
    output: np.ndarray
    direction: np.ndarray


def main():
    save_dir = "tmp_result"
    os.makedirs(os.path.join(save_dir, "replay"), exist_ok=True)

    video_file_name = "replay.mp4"
    result_file_name = "result.pkl"

    logger = Logger.load(os.path.join(save_dir, result_file_name))
    ind = logger[-1].min_ind

    settings = Settings()
    brain_builder = BrainBuilder(settings)
    task_generator = framework.TaskGenerator(settings, brain_builder)

    # task_generator.robot_positions = [
    #     (1.1262106310606397, -1.6779491283906323, 214.50974959400716)
    # ]
    # task_generator.food_positions = [
    #     (-1.810117866406546, 1.2444780212695203)
    # ]

    task = task_generator.generate(ind, debug=True)

    ind.dump = framework.entry_points.record(
        settings,
        os.path.join(save_dir, f"replay/{video_file_name}"),
        task,
    )

    robot_data = {f"robot{i}": [] for i in range(settings.Robot.NUM)}

    for delta in ind.dump.deltas:
        for i in range(settings.Robot.NUM):
            robot_data[f"robot{i}"].append(
                RobotInfo(
                    position=delta.robot_pos[f"robot{i}"],
                    input=delta.robot_inputs[f"robot{i}"],
                    output=delta.robot_outputs[f"robot{i}"],
                    direction=delta.robot_direction[f"robot{i}"],
                )
            )

    # 各ロボットごとにアニメーションを作成
    for i in range(settings.Robot.NUM):
        fig, ax = plt.subplots()
        ax.set_xlim(-settings.Simulation.WORLD_WIDTH / 2, settings.Simulation.WORLD_WIDTH / 2)
        ax.set_ylim(-settings.Simulation.WORLD_HEIGHT / 2, settings.Simulation.WORLD_HEIGHT / 2)
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        scatter = ax.plot([], [], 'o')[0]
        quiver = (
            ax.quiver([0], [0], [0], [1], angles='xy', scale_units='xy', scale=0.8, color='r'),  # Robot
            ax.quiver([0], [0], [0], [1], angles='xy', scale_units='xy', scale=0.2, color='g'),  # Food
            ax.quiver([0], [0], [0], [1], angles='xy', scale_units='xy', scale=1, color='b'),  # Nest
            ax.quiver([0], [0], [0], [1], angles='xy', scale_units='xy', scale=1, color='y'),  # direction
        )

        def update(frame, robot_id=f"robot{i}"):
            rd = robot_data[robot_id][frame]
            scatter.set_data([rd.position[0]], [rd.position[1]])

            quiver[3].set_offsets([rd.position[0], rd.position[1]])
            quiver[3].set_UVC(rd.direction[0], rd.direction[1])

            rotation_matrix = np.linalg.inv(np.array([
                [rd.direction[1], -rd.direction[0]],
                [rd.direction[0], rd.direction[1]]
            ]))

            for j, info in zip(range(3), [rd.input[0:2], rd.input[2:4], rd.input[4:6]]):
                rotated_vector = rotation_matrix @ np.array([info[0], info[1]])
                quiver[j].set_offsets([rd.position[0], rd.position[1]])
                quiver[j].set_UVC(rotated_vector[0], rotated_vector[1])

            return scatter, *quiver

        ani = FuncAnimation(fig, update, frames=len(ind.dump.deltas))

        ani.save(
            os.path.join(save_dir, "replay", f"robot{i}_replay.mp4"),
            writer=FFMpegWriter(
                fps=int(1 / settings.Simulation.TIME_STEP),
                metadata={"artist": "Matplotlib"},
                bitrate=1800
            )
        )

        plt.close(fig)


if __name__ == '__main__':
    main()
