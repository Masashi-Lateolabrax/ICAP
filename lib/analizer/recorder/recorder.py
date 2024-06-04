import os.path

import mujoco
import numpy as np
import cv2

from datetime import datetime
from mujoco_xml_generator.utils.dummy_geom import draw_dummy_geoms

from lib.optimizer import MjcTaskInterface


def recorder(
        task: MjcTaskInterface,
        width: int, height: int,
        camera: mujoco.MjvCamera,
        length: int,
        timestep: float,
        working_directory: str,
        max_geom: int = 10000
):
    time = datetime.now()
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    img = np.zeros((height, width, 3), dtype=np.uint8)
    renderer = mujoco.Renderer(task.get_model(), height, width, max_geom=max_geom)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        os.path.join(working_directory, f"./result_{timestamp}.mp4"),
        fourcc, int(1 / timestep), (width, height)
    )

    for t in range(length):
        if (datetime.now() - time).seconds > 1:
            time = datetime.now()
            print(f"{t}/{length} ({t / length * 100}%)")

        task.calc_step()
        renderer.update_scene(task.get_data(), camera)
        draw_dummy_geoms(task.get_dummies(), renderer)
        renderer.render(out=img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(img)

    print("Saving")
    writer.release()
    print("Finish")
