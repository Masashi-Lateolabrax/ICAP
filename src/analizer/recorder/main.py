import numpy as np
import mujoco
import cv2
from datetime import datetime

from src.settings import HyperParameters, TaskGenerator, Task
from src.analizer.utils import load_parameter


def recorder(
        task: Task,
        width: int, height: int,
        camera: mujoco.MjvCamera,
        length: int = int(HyperParameters.Simulator.EPISODE / HyperParameters.Simulator.TIMESTEP + 0.5),
):
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    img = np.zeros((height, width, 3), dtype=np.uint8)
    renderer = mujoco.Renderer(task.get_model(), height, width)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        f"./result_{timestamp}.mp4", fourcc, int(1 / HyperParameters.Simulator.TIMESTEP), (width, height)
    )

    for t in range(length):
        print(f"{t}/{length} ({t / length * 100}%)")
        task.calc_step()
        renderer.update_scene(task.get_data(), camera)
        renderer.render(out=img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(img)

    print("Saving")
    writer.release()
    print("Finish")


def main():
    task_generator = TaskGenerator()

    camera = mujoco.MjvCamera()
    camera.elevation = -90
    camera.distance = 25

    resolution = 150

    task = task_generator.generate(None)
    recorder(
        task=task,
        width=int(6 * resolution), height=int(9 * resolution),
        length=int(HyperParameters.Simulator.EPISODE / HyperParameters.Simulator.TIMESTEP + 0.5),
        camera=camera,
    )


if __name__ == '__main__':
    main()
