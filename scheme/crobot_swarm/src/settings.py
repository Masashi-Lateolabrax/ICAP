from framework import Settings
from loss import Loss

Settings.Simulation.RENDER_WIDTH = 800
Settings.Simulation.RENDER_HEIGHT = 800
Settings.Simulation.RENDER_ZOOM = 19
Settings.Simulation.MAX_GEOM = 1000

Settings.CMAES.GENERATION = 300
Settings.CMAES.POPULATION = 100
Settings.CMAES.MU = 50
Settings.CMAES.SIGMA = 0.1

Settings.Simulation.TIME_STEP = 0.01
Settings.Simulation.TIME_LENGTH = int(30 / Settings.Simulation.TIME_STEP)

Settings.CMAES.LOSS = Loss()

Settings.Simulation.WORLD_WIDTH = 15
Settings.Simulation.WORLD_HEIGHT = 15

Settings.Robot.THINK_INTERVAL = 0.1
Settings.Robot.FoodSensor.GAIN = 1

Settings.Robot.NUM = 50
Settings.Food.NUM = 4
