import mujoco
import numpy as np
from icecream import ic


def render(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        render_shape: tuple[int, int],
        max_geom: int,
        img_buf: np.ndarray,
        pos: tuple[float, float, float],
        lookat: tuple[float, float, float]
):
    camera = mujoco.MjvCamera()
    pos = np.array(pos)
    lookat = np.array(lookat)
    sub = pos - lookat

    camera.lookat[:] = lookat
    camera.distance = np.linalg.norm(sub)
    camera.azimuth = np.arctan2(
        sub[1], sub[0]
    ) * 180 / mujoco.mjPI + 180
    camera.elevation = -np.arcsin(
        sub[2] / camera.distance
    ) * 180 / mujoco.mjPI

    try:
        with mujoco.Renderer(
                model, width=render_shape[0], height=render_shape[1], max_geom=max_geom
        ) as renderer:
            renderer.update_scene(data, camera)
            renderer.render(out=img_buf)

    except Exception as e:
        ic("MuJoCo render error:", e)
        img_buf.fill(0)
