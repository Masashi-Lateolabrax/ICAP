import genesis as gs
import torch

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


def example_run():
    backend = GenesisBackend()
    for i in range(1000):
        backend.step()
