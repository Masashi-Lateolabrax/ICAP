import numpy as np

from framework.simulator.const import Settings, ROBOT_SIZE

from loss import Loss, calc_loss_sigma

Settings.Simulation.RENDER_WIDTH = 800
Settings.Simulation.RENDER_HEIGHT = 800
Settings.Simulation.RENDER_ZOOM = 19
Settings.Simulation.MAX_GEOM = 1000

Settings.CMAES.GENERATION = 100
Settings.CMAES.POPULATION = 100
Settings.CMAES.MU = 50
Settings.CMAES.SIGMA = 0.1

Settings.Simulation.TIME_STEP = 0.01
Settings.Simulation.TIME_LENGTH = int(30 / Settings.Simulation.TIME_STEP)

Settings.CMAES.LOSS = Loss()
Settings.CMAES.LOSS.sigma_nest_and_food = calc_loss_sigma(4, 0.01)
Settings.CMAES.LOSS.GAIN_NEST_AND_FOOD = 1
Settings.CMAES.LOSS.sigma_robot_and_food = calc_loss_sigma(1, 0.3)
Settings.CMAES.LOSS.GAIN_ROBOT_AND_FOOD = 0.5

Settings.Simulation.WORLD_WIDTH = 15
Settings.Simulation.WORLD_HEIGHT = 15

Settings.Robot.THINK_INTERVAL = 0.1
Settings.Robot.OtherRobotSensor.GAIN = (lambda d, v: (1 - v) / (v * d))(2, 0.1)
Settings.Robot.FoodSensor.GAIN = (lambda d, v: (1 - v) / (v * d))(4, 0.1)

Settings.Robot.NUM = 7 * 7
Settings.Food.NUM = 4


def set_positions(settings: Settings):
    settings.Robot.POSITION = []
    area = (np.sqrt(settings.Robot.NUM) - 1) * (ROBOT_SIZE + 0.05)
    for x in np.linspace(-area, area, int(np.sqrt(settings.Robot.NUM)), endpoint=True):
        for y in np.linspace(area, -area, int(np.sqrt(settings.Robot.NUM)), endpoint=True):
            settings.Robot.POSITION.append((x, y, 0))

    settings.Food.POSITION = [
        (0, 3.5),
        (3.5, 0),
        (0, -3.5),
        (-3.5, 0)
    ]
