import time

import numpy as np
import cv2

from framework.prelude import *

from src.analysis_mod.structure.debug_data import DebugData


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
        pos_[0] + int(length * direction[0]),
        pos_[1] - int(length * direction[1])
    )
    cv2.arrowedLine(img_, pos_, end, color, thickness, tipLength=tip_length)


def input_animation(settings: Settings, debug_data: list[DebugData], file_path: str):
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
    for i, di in enumerate(debug_data):
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
            print(f"Recording input animation frame {i + 1}/{len(debug_data)}")
            timer = time.time()

    writer.release()
