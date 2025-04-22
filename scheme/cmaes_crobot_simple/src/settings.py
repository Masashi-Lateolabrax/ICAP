from framework import Settings
from loss import Loss

Settings.Simulation.RENDER_WIDTH = 800
Settings.Simulation.RENDER_HEIGHT = 800
Settings.Simulation.RENDER_ZOOM = 10

Settings.CMAES.GENERATION = 500
Settings.CMAES.POPULATION = 1000
Settings.CMAES.MU = 500
Settings.CMAES.SIGMA = 0.01

Settings.Simulation.TIME_STEP = 0.01
Settings.Simulation.TIME_LENGTH = int(30 / Settings.Simulation.TIME_STEP)

Settings.CMAES.LOSS = Loss()

Settings.Simulation.WORLD_WIDTH = 8
Settings.Simulation.WORLD_HEIGHT = 8

Settings.Robot.THINK_INTERVAL = 0.1
Settings.Robot.FoodSensor.GAIN = 1

Settings.Robot.NUM = 1
Settings.Food.NUM = 1
