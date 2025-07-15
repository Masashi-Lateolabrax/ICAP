import dataclasses
import math
import os.path
import pickle
import time
import re

import numpy as np
import cv2

from framework.prelude import *

from client import MySettings, Simulator


@dataclasses.dataclass
class DebugInfo:
    time: float
    robot_positions: list[np.ndarray]
    food_positions: list[np.ndarray]
    food_directions: list[np.ndarray]


class SimulatorForDebugging(Simulator):
    def __init__(self, settings: Settings, parameters: Individual, render: bool = False):
        super().__init__(settings, parameters, render)

        self.timer = 0
        self.debug_data = []

    def step(self):
        super().step()

        self.timer += self.settings.Simulation.TIME_STEP
        self.debug_data.append(DebugInfo(
            time=self.timer,
            robot_positions=[np.copy(r.xpos) for r in self.robot_values],
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

        for robot_pos in di.robot_positions:
            pos = world_to_pixel(robot_pos)
            cv2.circle(buffer, pos, 5, (255, 0, 0), -1)
        for food_pos, food_dir in zip(di.food_positions, di.food_directions):
            pos = world_to_pixel(food_pos)
            cv2.circle(buffer, pos, 5, (0, 255, 0), -1)

            arrow_length = 20
            arrow_end = (
                pos[0] + int(arrow_length * food_dir[0]),
                pos[1] - int(arrow_length * food_dir[1])
            )
            cv2.arrowedLine(buffer, pos, arrow_end, (0, 0, 255), 2, tipLength=0.3)

        img = cv2.cvtColor(buffer, cv2.COLOR_RGB2BGR)
        writer.write(img)

        if time.time() - timer > 1.0:
            print(f"Recording input animation frame {i + 1}/{len(debug_info)}")
            timer = time.time()

    writer.release()


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


if __name__ == '__main__':
    main()
