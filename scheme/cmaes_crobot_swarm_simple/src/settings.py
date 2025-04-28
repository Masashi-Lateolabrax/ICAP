import numpy as np

from framework.simulator.const import Settings, ROBOT_SIZE

from loss import Loss, calc_loss_sigma

## Rendering settings
Settings.Simulation.Render.RENDER_WIDTH = 800
Settings.Simulation.Render.RENDER_HEIGHT = 900
Settings.Simulation.Render.RENDER_ZOOM = 25
Settings.Simulation.Render.MAX_GEOM = 1000

## Light settings
Settings.Simulation.Render.LIGHT_AMBIENT = (1, 1, 1)
Settings.Simulation.Render.LIGHT_DIFFUSE = (1, 1, 1)
Settings.Simulation.Render.LIGHT_SPECULAR = (1, 1, 1)

## World Size
Settings.Simulation.WORLD_WIDTH = 16
Settings.Simulation.WORLD_HEIGHT = 18

## Optimization settings
Settings.CMAES.GENERATION = 500
Settings.CMAES.POPULATION = 100
Settings.CMAES.MU = 50
Settings.CMAES.SIGMA = 0.3

## Simulation settings
Settings.Simulation.TIME_STEP = 0.01
Settings.Simulation.TIME_LENGTH = int(45 / Settings.Simulation.TIME_STEP)

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
Settings.Robot.NUM = 3 * 3
Settings.Food.NUM = 2


def set_positions(settings: Settings):
    settings.Robot.POSITION = []
    area = (np.sqrt(settings.Robot.NUM) - 1) * (ROBOT_SIZE + 0.05)
    for x in np.linspace(-area, area, int(np.sqrt(settings.Robot.NUM)), endpoint=True):
        for y in np.linspace(area, -area, int(np.sqrt(settings.Robot.NUM)), endpoint=True):
            settings.Robot.POSITION.append((x, y, 0))

    settings.Food.POSITION = [
        (0, -6),
        (0, 6)
    ]


def randomize_direction(settings: Settings, sigma: float = 0.1):
    rs = np.random.uniform(
        low=-sigma * 180,
        high=sigma * 180,
        size=len(settings.Robot.POSITION)
    )
    for i, r in enumerate(rs):
        settings.Robot.POSITION[i] = (
            settings.Robot.POSITION[i][0],
            settings.Robot.POSITION[i][1],
            90 + r
        )


def randomize_food_position(settings: Settings):
    settings.Food.POSITION = [
        (0, np.random.uniform(low=-7, high=-3)),
        (0, np.random.uniform(low=3, high=7))
    ]
