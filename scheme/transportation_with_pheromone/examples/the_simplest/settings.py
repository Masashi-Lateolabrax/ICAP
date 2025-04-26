from prelude import *


class Settings(framework.Settings):
    def __init__(self, loss: framework.interfaceis.Loss):
        super().__init__()

        self.CMAES.LOSS = loss

        self.Simulation.RENDER_HEIGHT = 800
        self.Simulation.RENDER_WIDTH = 800

