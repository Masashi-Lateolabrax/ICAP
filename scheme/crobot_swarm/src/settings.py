import numpy as np

from framework.simulator.const import Settings, ROBOT_SIZE

from loss import Loss, calc_loss_sigma

## Rendering settings
Settings.Simulation.Render.RENDER_WIDTH = 800
Settings.Simulation.Render.RENDER_HEIGHT = 800
Settings.Simulation.Render.RENDER_ZOOM = 19
Settings.Simulation.Render.MAX_GEOM = 1000

## World Size
Settings.Simulation.WORLD_WIDTH = 15
Settings.Simulation.WORLD_HEIGHT = 15

## Optimization settings
Settings.CMAES.GENERATION = 300
Settings.CMAES.POPULATION = 100
Settings.CMAES.MU = 50
Settings.CMAES.SIGMA = 0.1

## Simulation settings
Settings.Simulation.TIME_STEP = 0.01
Settings.Simulation.TIME_LENGTH = int(30 / Settings.Simulation.TIME_STEP)

## Loss settings
Settings.CMAES.LOSS = Loss()
### Nest and Food
Settings.CMAES.LOSS.sigma_nest_and_food = calc_loss_sigma(4, 0.01)
Settings.CMAES.LOSS.GAIN_NEST_AND_FOOD = 1
### Robot and Food
Settings.CMAES.LOSS.sigma_robot_and_food = calc_loss_sigma(1, 0.3)
Settings.CMAES.LOSS.GAIN_ROBOT_AND_FOOD = 0.1

## Robot Settings
Settings.Robot.THINK_INTERVAL = 0.3
Settings.Robot.OtherRobotSensor.GAIN = (lambda d, v: (1 - v) / (v * d))(2, 0.1)
Settings.Robot.FoodSensor.GAIN = (lambda d, v: (1 - v) / (v * d))(4, 0.1)

## Food Settings
Settings.Food.REPLACEMENT = True

## Number of Robots and Food
Settings.Robot.NUM = 49
Settings.Food.NUM = 4
