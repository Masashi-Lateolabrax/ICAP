import framework
from loss import Loss


class Settings(framework.Settings):
    def __init__(self):
        super().__init__()

        self.Simulation.RENDER_HEIGHT = 800
        self.Simulation.RENDER_WIDTH = 800

        self.CMAES.GENERATION = 300
        self.CMAES.POPULATION = 1000
        self.CMAES.MU = int(self.CMAES.POPULATION / 2)
        self.CMAES.SIGMA = 0.1

        self.Simulation.TIME_STEP = 0.01
        self.Simulation.TIME_LENGTH = int(30 / self.Simulation.TIME_STEP)

        self.CMAES.LOSS = Loss()

        self.Simulation.WORLD_WIDTH = 8
        self.Simulation.WORLD_HEIGHT = 8

        self.Robot.THINK_INTERVAL = 1
        self.Robot.FoodSensor.GAIN = 1

    def __getstate__(self):
        state = self.__dict__.copy()
        state['CMAES.LOSS'] = self.CMAES.LOSS.__getstate__()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.CMAES.LOSS = Loss()
        self.CMAES.LOSS.__setstate__(state['CMAES.LOSS'])
