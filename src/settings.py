import numpy as np
from framework.prelude import Settings, calc_loss_sigma, Position, RobotLocation


class MySettings(Settings):
    """
    Global settings for the optimization framework.
    This class holds all the configuration parameters used throughout the application.
    """
    pass


###################################################################################
# Server settings
###################################################################################
MySettings.Server.HOST = "localhost"
MySettings.Server.PORT = 5000
MySettings.Server.SOCKET_BACKLOG = 10
MySettings.Server.TIMEOUT = 5.0
###################################################################################

###################################################################################
# Render settings
###################################################################################
MySettings.Render.RENDER_WIDTH = 100
MySettings.Render.RENDER_HEIGHT = 100
MySettings.Render.LIGHT_AMBIENT = 1.0
MySettings.Render.LIGHT_DIFFUSE = 1.0
MySettings.Render.LIGHT_SPECULAR = 1.0

###################################################################################
# Optimization settings
###################################################################################
MySettings.Optimization.dimension = None  # Will be set later based on the neural network dimension
MySettings.Optimization.population_size = 20
MySettings.Optimization.generations = 1000
MySettings.Optimization.sigma = 0.5
###################################################################################


###################################################################################
# Loss settings
###################################################################################
MySettings.Loss.OFFSET_NEST_AND_FOOD = 0
MySettings.Loss.SIGMA_NEST_AND_FOOD = calc_loss_sigma(4, 0.01)
MySettings.Loss.GAIN_NEST_AND_FOOD = 1
MySettings.Loss.OFFSET_ROBOT_AND_FOOD = Settings.Robot.RADIUS + Settings.Food.RADIUS
MySettings.Loss.SIGMA_ROBOT_AND_FOOD = calc_loss_sigma(1, 0.3)
MySettings.Loss.GAIN_ROBOT_AND_FOOD = 0.01
MySettings.Loss.REGULARIZATION_COEFFICIENT = 0
###################################################################################


###################################################################################
# Simulation settings
###################################################################################
MySettings.Simulation.TIME_STEP = 0.01
MySettings.Simulation.TIME_LENGTH = 60
MySettings.Simulation.WORLD_WIDTH = 10.0
MySettings.Simulation.WORLD_HEIGHT = 10.0
MySettings.Simulation.WALL_THICKNESS = 1
MySettings.Simulation.WALL_HEIGHT = 1
###################################################################################


###################################################################################
# Robot settings
###################################################################################
MySettings.Robot.HEIGHT = 0.1
MySettings.Robot.RADIUS = 0.175
MySettings.Robot.DISTANCE_BETWEEN_WHEELS = 0.175 * 2 * 0.8
MySettings.Robot.MAX_SPEED = 0.8
MySettings.Robot.COLOR = (1, 1, 0, 1)
MySettings.Robot.MASS = 10
MySettings.Robot.ACTUATOR_MOVE_KV = 100
MySettings.Robot.ACTUATOR_ROT_KV = 10
MySettings.Robot.ROBOT_SENSOR_GAIN = 1.0
MySettings.Robot.FOOD_SENSOR_GAIN = 1.0
MySettings.Robot.NUM = 1
MySettings.Robot.INITIAL_POSITION = [
    RobotLocation(0, 0, np.pi / 2),
]
###################################################################################


###################################################################################
# Food settings
###################################################################################
MySettings.Food.RADIUS = 0.5
MySettings.Food.DENSITY = 80
MySettings.Food.HEIGHT = 0.07
MySettings.Food.COLOR = (0, 1, 1, 1)
MySettings.Food.NUM = 1
MySettings.Food.INITIAL_POSITION = [
    Position(0, 2),
]
###################################################################################


###################################################################################
# Nest settings
###################################################################################
MySettings.Nest.POSITION = Position(0.0, 0.0)
MySettings.Nest.RADIUS = 1.0
MySettings.Nest.HEIGHT = 0.01
MySettings.Nest.COLOR = (0, 1, 0, 1)
###################################################################################
