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


def interpolate_color(normalized_distance: float) -> tuple[int, int, int]:
    """
    Convert normalized distance [0,1] to RGB color.
    0.0 (far) -> green (0, 255, 0)
    1.0 (close) -> red (255, 0, 0)
    """
    # Clamp to [0, 1]
    t = np.clip(normalized_distance, 0.0, 1.0)

    # Linear interpolation from green to red
    r = int(255 * t)
    g = int(255 * (1.0 - t))
    b = 0

    return (r, g, b)


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

    # Get sensor configuration
    num_rays = settings.Robot.DEPTH_SENSOR_NUM_RAYS
    max_range = settings.Robot.DEPTH_SENSOR_MAX_RANGE
    offset = settings.Robot.RADIUS

    timer = time.time()
    for i, di in enumerate(debug_data):
        buffer.fill(255)

        for robot_pos, robot_dir, inputs in zip(di.robot_positions, di.robot_directions, di.robot_inputs):
            pos = world_to_pixel(robot_pos)
            cv2.circle(buffer, pos, 5, (200, 0, 0), -1)

            # Draw the robot direction
            draw_arrowed_line(buffer, pos, robot_dir, 20, (200, 0, 0), thickness=2, tip_length=0.3)

            # Draw the nest direction by robot sight (DirectionSensor output)
            nest_direction = rotate_vector_2d(robot_dir, inputs[1] * np.pi)
            draw_arrowed_line(buffer, pos, nest_direction, 20, (255, 100, 0), thickness=2, tip_length=0.3)

            # Draw DepthSensor rays
            robot_yaw = np.arctan2(robot_dir[1], robot_dir[0])
            for ray_idx in range(num_rays):
                # Get normalized distance [0,1]: 1=close, 0=far
                normalized_distance = inputs[5 + ray_idx]

                # Convert to actual distance from robot surface
                surface_distance = (1.0 - normalized_distance) * max_range

                # Add offset to get total distance from robot center
                total_distance = surface_distance + offset

                # Calculate ray angle (evenly distributed 360 degrees)
                ray_angle = 2 * np.pi * ray_idx / num_rays

                # Calculate absolute direction in world coordinates
                absolute_angle = robot_yaw + ray_angle
                ray_direction = np.array([np.cos(absolute_angle), np.sin(absolute_angle)])

                # Calculate ray endpoint in world coordinates
                ray_end_world = robot_pos + total_distance * ray_direction

                # Convert to pixel coordinates
                ray_end_pixel = world_to_pixel(ray_end_world)

                # Get color based on normalized distance
                color = interpolate_color(normalized_distance)

                # Draw ray line from robot center to intersection point
                cv2.line(buffer, pos, ray_end_pixel, color, 1)

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
