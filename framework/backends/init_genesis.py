import genesis as gs
import torch
import numpy as np

from icecream import ic
from ..interfaces import SimulatorBackend

_genesis_initialized = False

if not _genesis_initialized:
    if torch.cuda.is_available():
        gs.init(backend=gs.gs_backend.gpu)
    else:
        gs.init(backend=gs.gs_backend.cpu)
    _genesis_initialized = True


class GenesisBackend(SimulatorBackend):
    def __init__(self):
        self.scene = gs.Scene(show_viewer=True)
        self.scene.add_entity(gs.morphs.Plane())
        self.scene.build()

    def step(self):
        self.scene.step()

    def render(self, img_buf: np.ndarray, pos: tuple[float, float, float], lookat: tuple[float, float, float]):
        try:
            rendered_img = self.scene.render(camera_pos=pos, camera_lookat=lookat)
            if rendered_img is not None and img_buf is not None:
                h, w = img_buf.shape[:2]
                if rendered_img.shape[:2] == (h, w):
                    img_buf[:] = rendered_img
                else:
                    import cv2
                    resized = cv2.resize(rendered_img, (w, h))
                    img_buf[:] = resized
        except Exception as e:
            ic("Genesis render error:", e)
            if img_buf is not None:
                img_buf.fill(0)


def example_run():
    backend = GenesisBackend()
    for i in range(1000):
        backend.step()
