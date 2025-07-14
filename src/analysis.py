import math
import os.path
import time
import re

import numpy as np
import cv2

from framework.prelude import *

from client import MySettings, Simulator


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


def record(settings: Settings, parameters: Individual, file_path: str):
    buffer = np.zeros(
        (
            settings.Render.RENDER_HEIGHT,
            settings.Render.RENDER_WIDTH,
            3
        ),
        dtype=np.uint8
    )

    simulator = Simulator(settings, parameters, render=True)

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


def main():
    settings = MySettings()

    save_dir = os.path.abspath(
        get_latest_folder(settings.Storage.SAVE_DIRECTORY)
    )
    saved_individual = SavedIndividual.load(
        latest_saved_individual_file(save_dir)
    )

    best_individual: Individual = min(saved_individual.individuals, key=lambda i: i.get_fitness())

    record(
        settings,
        best_individual,
        os.path.join(save_dir, "movie.mp4")
    )


if __name__ == '__main__':
    main()
