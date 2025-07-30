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
###################################################################################


###################################################################################
# Render settings
###################################################################################
MySettings.Render.RENDER_WIDTH = 500
MySettings.Render.RENDER_HEIGHT = 500

MySettings.Render.LIGHT_AMBIENT = 1.0
MySettings.Render.LIGHT_DIFFUSE = 1.0
MySettings.Render.LIGHT_SPECULAR = 1.0

MySettings.Render.CAMERA_POS = (0.0, -1e-3, 13.0)
MySettings.Render.CAMERA_LOOKAT = (0.0, 0.0, 0.0)
###################################################################################


###################################################################################
# Optimization settings
###################################################################################
MySettings.Optimization.DIMENSION = None  # Will be set later based on the neural network dimension
MySettings.Optimization.POPULATION = 300
MySettings.Optimization.GENERATION = 1000
MySettings.Optimization.SIGMA = 0.1
MySettings.Optimization.CLIP = (-1.0, 1.0)
###################################################################################


###################################################################################
# Robot settings
###################################################################################
MySettings.Robot.HEIGHT = 0.1
MySettings.Robot.RADIUS = 0.175
MySettings.Robot.DISTANCE_BETWEEN_WHEELS = 0.175 * 2 * 0.8
MySettings.Robot.MAX_SPEED = 0.8
MySettings.Robot.MASS = 10

MySettings.Robot.COLOR = (1, 1, 0, 1)

MySettings.Robot.THINK_INTERVAL = 0.01

MySettings.Robot.ACTUATOR_MOVE_KV = 100
MySettings.Robot.ACTUATOR_ROT_KV = 10

MySettings.Robot.ROBOT_SENSOR_GAIN = 1.0
MySettings.Robot.FOOD_SENSOR_GAIN = 1.0

MySettings.Robot.NUM = 9
MySettings.Robot.INITIAL_POSITION = [
    RobotLocation(-0.175 * 2 - 0.1, 0.5, np.pi),
    RobotLocation(0, 0.5, np.pi),
    RobotLocation(0.175 * 2 + 0.1, 0.5, np.pi),

    RobotLocation(-0.175 * 2 - 0.1, 0, np.pi),
    RobotLocation(0, 0, np.pi),
    RobotLocation(0.175 * 2 + 0.1, 0, np.pi),

    RobotLocation(-0.175 * 2 - 0.1, -0.5, np.pi),
    RobotLocation(0, -0.5, np.pi),
    RobotLocation(0.175 * 2 + 0.1, -0.5, np.pi),
]
###################################################################################


###################################################################################
# Food settings
###################################################################################
MySettings.Food.RADIUS = 0.5
MySettings.Food.HEIGHT = 0.07

MySettings.Food.DENSITY = 80
MySettings.Food.COLOR = (0, 1, 1, 1)

MySettings.Food.NUM = 2
MySettings.Food.INITIAL_POSITION = [
    Position(2, 2),
    Position(-2, 2),
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


###################################################################################
# Loss settings
###################################################################################
MySettings.Loss.OFFSET_NEST_AND_FOOD = 0
MySettings.Loss.SIGMA_NEST_AND_FOOD = calc_loss_sigma(4, 0.01)
MySettings.Loss.GAIN_NEST_AND_FOOD = 1

MySettings.Loss.OFFSET_ROBOT_AND_FOOD = Settings.Robot.RADIUS + Settings.Food.RADIUS
MySettings.Loss.SIGMA_ROBOT_AND_FOOD = calc_loss_sigma(1, 0.3)
MySettings.Loss.GAIN_ROBOT_AND_FOOD = 0.01

MySettings.Loss.REGULARIZATION_COEFFICIENT = 1e-3
###################################################################################


###################################################################################
# Simulation settings
###################################################################################
MySettings.Simulation.TIME_STEP = 0.01
MySettings.Simulation.TIME_LENGTH = 90

MySettings.Simulation.WORLD_WIDTH = 10.0
MySettings.Simulation.WORLD_HEIGHT = 10.0

MySettings.Simulation.WALL_THICKNESS = 1
MySettings.Simulation.WALL_HEIGHT = 1
###################################################################################


###################################################################################
# Storage settings
###################################################################################
MySettings.Storage.SAVE_INDIVIDUALS = True
MySettings.Storage.SAVE_DIRECTORY = "results"
MySettings.Storage.SAVE_INTERVAL = 1
MySettings.Storage.TOP_N = 5
###################################################################################
