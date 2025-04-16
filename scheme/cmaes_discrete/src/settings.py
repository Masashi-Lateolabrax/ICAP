from framework import Settings
from loss import Loss

Settings.Simulation.RENDER_WIDTH = 800
Settings.Simulation.RENDER_HEIGHT = 800
Settings.Simulation.RENDER_ZOOM = 10

Settings.CMAES.GENERATION = 500
Settings.CMAES.POPULATION = 1000
Settings.CMAES.MU = 100
Settings.CMAES.SIGMA = 0.1

Settings.Simulation.TIME_STEP = 0.01
Settings.Simulation.TIME_LENGTH = int(30 / Settings.Simulation.TIME_STEP)

Settings.CMAES.LOSS = Loss()
Settings.CMAES.LOSS.GAIN_NEST_AND_FOOD = 1
Settings.CMAES.LOSS.GAIN_ROBOT_AND_FOOD = 0.1

Settings.Simulation.WORLD_WIDTH = 8
Settings.Simulation.WORLD_HEIGHT = 8

Settings.Robot.THINK_INTERVAL = 0.5
Settings.Robot.ARGMAX_SELECTION = False
Settings.Robot.FoodSensor.GAIN = 1

Settings.Robot.NUM = 1
Settings.Food.NUM = 1
