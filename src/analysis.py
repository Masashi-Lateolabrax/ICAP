import dataclasses
import math
import os.path
import pickle
import time
import re

import numpy as np
import cv2
import matplotlib.pyplot as plt

from framework.prelude import *

from client import MySettings, Simulator


@dataclasses.dataclass
class DebugInfo:
    time: float
    robot_positions: list[np.ndarray]
    robot_inputs: np.ndarray
    robot_directions: list[np.ndarray]
    food_positions: list[np.ndarray]
    food_directions: list[np.ndarray]


class SimulatorForDebugging(Simulator):
    def __init__(self, settings: Settings, parameters: Individual, render: bool = False):
        super().__init__(settings, parameters, render)

        self.step_time = 0
        self.debug_data = []

    def step(self):
        super().step()

        self.step_time += self.settings.Simulation.TIME_STEP
        self.debug_data.append(DebugInfo(
            time=self.step_time,
            robot_positions=[np.copy(r.xpos) for r in self.robot_values],
            robot_inputs=self.input_ndarray.copy(),
            robot_directions=[np.copy(r.xdirection) for r in self.robot_values],
            food_positions=[np.copy(f.xpos) for f in self.food_values],
            food_directions=[np.copy(f.direction) for f in self.food_values]
        ))


def get_latest_folder(base_directory: str) -> str:
    folders = [f for f in os.listdir(base_directory)
               if os.path.isdir(os.path.join(base_directory, f))]

    if not folders:
        raise ValueError(f"No folders found in {base_directory}")

    folders.sort(reverse=True)

    return os.path.join(base_directory, folders[0])


def latest_saved_individual_file(directory: str) -> str:
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Extract numeric parts from filenames matching pattern like "generation_0000.pkl"
    pattern = re.compile(r'generation_(\d+)\.pkl')
    max_num = -1
    latest_file_name = None

    for filename in files:
        match = pattern.match(filename)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
                latest_file_name = filename

    if latest_file_name:
        return os.path.join(directory, latest_file_name)
    else:
        raise FileNotFoundError("No generation files found in directory")


def record(settings: Settings, parameters: Individual, file_path: str) -> list[DebugInfo]:
    buffer = np.zeros(
        (
            settings.Render.RENDER_HEIGHT,
            settings.Render.RENDER_WIDTH,
            3
        ),
        dtype=np.uint8
    )

    simulator = SimulatorForDebugging(settings, parameters, render=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        file_path,
        fourcc, int(1 / settings.Simulation.TIME_STEP),
        (settings.Render.RENDER_WIDTH, settings.Render.RENDER_HEIGHT)
    )

    timer = time.time()
    length = math.floor(settings.Simulation.TIME_LENGTH / settings.Simulation.TIME_STEP)
    for t in range(length):
        simulator.step()
        simulator.render(buffer, settings.Render.CAMERA_POS, settings.Render.CAMERA_LOOKAT)
        img = cv2.cvtColor(buffer, cv2.COLOR_RGB2BGR)
        writer.write(img)

        if time.time() - timer > 1.0:
            print(f"Recording frame {t + 1}/{length}")
            timer = time.time()

    writer.release()

    return simulator.debug_data


def input_animation(settings: Settings, debug_info: list[DebugInfo], file_path: str):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        file_path,
        fourcc, int(1 / settings.Simulation.TIME_STEP),
        (settings.Render.RENDER_WIDTH, settings.Render.RENDER_HEIGHT)
    )

    def world_to_pixel(world_pos: np.ndarray) -> tuple[int, int]:
        world_half = np.array([settings.Simulation.WORLD_WIDTH, settings.Simulation.WORLD_HEIGHT]) / 2.0
        world_size = np.array([settings.Simulation.WORLD_WIDTH, settings.Simulation.WORLD_HEIGHT])
        render_size = np.array([settings.Render.RENDER_WIDTH, settings.Render.RENDER_HEIGHT])

        normalized = (world_pos[:2] + world_half) / world_size
        pixel_pos = normalized * render_size
        pixel_pos[1] = render_size[1] - pixel_pos[1]

        pixel_pos = np.clip(pixel_pos, 0, render_size - 1)
        return int(pixel_pos[0]), int(pixel_pos[1])

    def rotate_vector_2d(vector, angle_rad):
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        rotation = np.array([[c, -s], [s, c]])
        return rotation @ vector

    def draw_arrowed_line(
            img_: np.ndarray, pos_: tuple[int, int], direction, length,
            color: tuple[int, int, int], thickness=1, tip_length=0.1
    ):
        end = (
            pos[0] + int(length * direction[0]),
            pos[1] - int(length * direction[1])
        )
        cv2.arrowedLine(img_, pos_, end, color, thickness, tipLength=tip_length)

    buffer = np.zeros(
        (
            settings.Render.RENDER_HEIGHT,
            settings.Render.RENDER_WIDTH,
            3
        ),
        dtype=np.uint8
    )

    timer = time.time()
    for i, di in enumerate(debug_info):
        buffer.fill(255)

        for robot_pos, robot_dir, inputs in zip(di.robot_positions, di.robot_directions, di.robot_inputs):
            pos = world_to_pixel(robot_pos)
            cv2.circle(buffer, pos, 5, (200, 0, 0), -1)

            # Draw the robot direction
            draw_arrowed_line(buffer, pos, robot_dir, 20, (200, 0, 0), thickness=2, tip_length=0.3)

            # Draw the nest direction by robot sight.
            nest_direction = rotate_vector_2d(robot_dir, inputs[5] * np.pi)
            draw_arrowed_line(buffer, pos, nest_direction, 20, (255, 100, 0), thickness=2, tip_length=0.3)

            # Draw the food direction
            food_direction = rotate_vector_2d(robot_dir, inputs[3] * np.pi)
            draw_arrowed_line(buffer, pos, food_direction, 20, (255, 0, 100), thickness=2, tip_length=0.3)

            # Draw the other robot direction
            food_direction = rotate_vector_2d(robot_dir, inputs[1] * np.pi)
            draw_arrowed_line(buffer, pos, food_direction, 20, (255, 100, 100), thickness=2, tip_length=0.3)

        for food_pos, food_dir in zip(di.food_positions, di.food_directions):
            pos = world_to_pixel(food_pos)
            cv2.circle(buffer, pos, 5, (0, 200, 0), -1)

            # Draw the food direction
            draw_arrowed_line(buffer, pos, food_dir, 20, (0, 200, 0), thickness=2, tip_length=0.3)

        img = cv2.cvtColor(buffer, cv2.COLOR_RGB2BGR)
        writer.write(img)

        if time.time() - timer > 1.0:
            print(f"Recording input animation frame {i + 1}/{len(debug_info)}")
            timer = time.time()

    writer.release()


def plot_best_fitness(saved_individuals: IndividualRecorder, filepath: str):
    generations = []
    fitnesses = []

    for rec in saved_individuals:
        generations.append(rec.generation)
        fitnesses.append(rec.best_fitness)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(generations, fitnesses)

    ax.set_title('Best Fitness Over Generations')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Fitness')
    ax.grid()

    plt.savefig(filepath)


def plot_max_value_in_parameters(saved_individuals: IndividualRecorder, filepath: str, top_n=1):
    generations = np.arange(len(saved_individuals))
    values = np.zeros((top_n, len(saved_individuals)), dtype=float)
    index = np.zeros((top_n, len(saved_individuals)), dtype=int)

    for i, rec in enumerate(saved_individuals):
        abs_best_individual = [(i, abs(v), v) for i, v in enumerate(rec.best_individual)]
        sored_parameters = sorted(abs_best_individual, key=lambda x: x[1], reverse=True)

        for j in range(min(top_n, len(sored_parameters))):
            values[j, i] = sored_parameters[j][2]
            index[j, i] = sored_parameters[j][0]

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()

    for i in range(top_n):
        if i < values.shape[0]:
            ax1.plot(generations, values[i], label=f'Value {i}')
            ax2.scatter(generations, index[i], label=f'Index {i}')

    plt.savefig(filepath)


def parameter_heatmap(saved_individuals: IndividualRecorder, filepath: str):
    generations = np.arange(len(saved_individuals))
    num_parameters = len(saved_individuals[0].best_individual)

    heatmap_data = np.zeros((num_parameters, len(saved_individuals)), dtype=float)

    for i, rec in enumerate(saved_individuals):
        best_ind = rec.best_individual
        max_value = np.max(best_ind)
        for j in range(num_parameters):
            heatmap_data[j, i] = best_ind[j] / max_value

    fig, ax = plt.subplots()
    cax = ax.imshow(heatmap_data, aspect='auto', cmap='hot', interpolation='nearest')

    ax.set_title('Parameter Heatmap Over Generations')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Parameter Index')
    ax.set_xticks(np.arange(len(generations)))
    ax.set_yticks(np.arange(num_parameters))
    ax.set_xticklabels(generations)
    ax.set_yticklabels(np.arange(num_parameters))

    fig.colorbar(cax)
    plt.savefig(filepath)


def main():
    settings = MySettings()

    save_dir = os.path.abspath(
        get_latest_folder(settings.Storage.SAVE_DIRECTORY)
    )
    saved_individuals = IndividualRecorder.load(
        os.path.join(save_dir, "optimization_log.pkl")
    )
    debug_info_path = os.path.join(save_dir, "debug_info.pkl")

    rec: Rec = saved_individuals.get_best_rec()
    individual: Individual = rec.best_individual

    # Record the behavior of the best individual.
    # And save the debug info while doing so.
    file_path = os.path.join(save_dir, f"generation_{rec.generation}.mp4")
    if not os.path.exists(file_path) or not os.path.exists(debug_info_path):
        debug_info: list[DebugInfo] = record(
            settings,
            individual,
            file_path
        )
        with open(debug_info_path, 'wb') as f:
            pickle.dump(debug_info, f)
    else:
        with open(debug_info_path, 'rb') as f:
            debug_info: list[DebugInfo] = pickle.load(f)
        if not isinstance(debug_info, list):
            raise ValueError(f"Expected debug_info to be a list, got {type(debug_info)}")
        if not all(isinstance(di, DebugInfo) for di in debug_info):
            raise ValueError("All items in debug_info must be DebugInfo instances")

    # Create an input animation from the debug info.
    file_path = os.path.join(save_dir, f"input_animation_{rec.generation}.mp4")
    if not os.path.exists(file_path):
        input_animation(
            settings,
            debug_info,
            file_path
        )

    # Save the fitness plot.
    file_path = os.path.join(save_dir, "fitness_plot.png")
    if not os.path.exists(file_path):
        plot_best_fitness(saved_individuals, file_path)

    # Plot the maximum values in parameters.
    file_path = os.path.join(save_dir, "max_value_in_parameters.png")
    if not os.path.exists(file_path):
        plot_max_value_in_parameters(saved_individuals, file_path, 1)

    # Create a heatmap of the parameters.
    file_path = os.path.join(save_dir, "parameter_heatmap.png")
    if not os.path.exists(file_path):
        parameter_heatmap(saved_individuals, file_path)


if __name__ == '__main__':
    main()
