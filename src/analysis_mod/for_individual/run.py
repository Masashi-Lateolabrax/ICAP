import time
import math

import numpy as np
import cv2

from framework.prelude import *

from src.analysis_mod.structure.sim_interface import SimulatorForDebugInterface
from src.analysis_mod.structure.debug_data import DebugData


def run(
        settings: Settings, simulator: SimulatorForDebugInterface, video_writer: cv2.VideoWriter = None
) -> list[DebugData]:
    buffer_shape = (0,) if video_writer is None else (settings.Render.RENDER_HEIGHT, settings.Render.RENDER_WIDTH, 3)
    buffer = np.zeros(buffer_shape, dtype=np.uint8)

    timer = time.time()
    length = math.floor(settings.Simulation.TIME_LENGTH / settings.Simulation.TIME_STEP)
    for t in range(length):
        simulator.step()

        if video_writer is not None:
            simulator.render(buffer, settings.Render.CAMERA_POS, settings.Render.CAMERA_LOOKAT)
            img = cv2.cvtColor(buffer, cv2.COLOR_RGB2BGR)
            video_writer.write(img)

        if time.time() - timer > 1.0:
            print(f"Running frame {t + 1}/{length} (Recording: {str(video_writer is not None)})")
            timer = time.time()

    return simulator.debug_data()


def record(settings: Settings, simulator: SimulatorForDebugInterface, file_path: str) -> list[DebugData]:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        file_path,
        fourcc, int(1 / settings.Simulation.TIME_STEP),
        (settings.Render.RENDER_WIDTH, settings.Render.RENDER_HEIGHT)
    )
    debug_data = run(settings, simulator, writer)
    writer.release()
    return debug_data
