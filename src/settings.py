import numpy as np
from framework.prelude import Settings, calc_loss_sigma, Position, RobotLocation, ClippingFunctions, ETHANOL


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

MySettings.Render.MAX_GEOM = 11000
MySettings.Render.MAX_PHEROMONE = 1.0
###################################################################################


###################################################################################
# Optimization settings
###################################################################################
MySettings.Optimization.DIMENSION = None  # Will be set later based on the neural network dimension
MySettings.Optimization.POPULATION = 100
MySettings.Optimization.GENERATION = 1000
MySettings.Optimization.SIGMA = 1

MySettings.Optimization.CLIP = ClippingFunctions.none
###################################################################################


###################################################################################
# Robot settings
###################################################################################
MySettings.Robot.HEIGHT = 0.1
MySettings.Robot.RADIUS = 0.175
MySettings.Robot.DISTANCE_BETWEEN_WHEELS = 0.175 * 2 * 0.8
MySettings.Robot.MAX_SPEED = 0.8
MySettings.Robot.MAX_PHEROMONE_SECRETION = 0.05
MySettings.Robot.MASS = 10

MySettings.Robot.COLOR = (1, 1, 0, 1)

MySettings.Robot.THINK_INTERVAL = 0.01

MySettings.Robot.ACTUATOR_MOVE_KV = 100
MySettings.Robot.ACTUATOR_ROT_KV = 10

MySettings.Robot.ROBOT_SENSOR_GAIN = 1.0
MySettings.Robot.FOOD_SENSOR_GAIN = 1.0

MySettings.Robot.DEPTH_SENSOR_NUM_RAYS = 16
MySettings.Robot.DEPTH_SENSOR_MAX_RANGE = 1.0

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
MySettings.Loss.OFFSET_FOOD_AND_NEST = 0
MySettings.Loss.SIGMA_FOOD_AND_NEST = calc_loss_sigma(4, 0.01)
MySettings.Loss.GAIN_FOOD_AND_NEST = 1

MySettings.Loss.OFFSET_FOOD_AND_ROBOT = Settings.Robot.RADIUS + Settings.Food.RADIUS
MySettings.Loss.SIGMA_FOOD_AND_ROBOT = calc_loss_sigma(1, 0.3)
MySettings.Loss.GAIN_FOOD_AND_ROBOT = 0.01

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

MySettings.Simulation.TEMPERATURE = 300.0
MySettings.Simulation.GRAVITY = (0.0, 0.0, -9.81)
###################################################################################


###################################################################################
# Pheromone settings
###################################################################################
MySettings.Pheromone.ACTIVE = True

MySettings.Pheromone.CELL_SIZE = 0.1
MySettings.Pheromone.CELL_SIZE_Z = 0.1
MySettings.Pheromone.WIDTH_NUM = int(MySettings.Simulation.WORLD_WIDTH / MySettings.Pheromone.CELL_SIZE)
MySettings.Pheromone.HEIGHT_NUM = int(MySettings.Simulation.WORLD_HEIGHT / MySettings.Pheromone.CELL_SIZE)

MySettings.Pheromone.ITERATIONS_PER_STEP = 1

MySettings.Pheromone.TEMPERATURE = 300
MySettings.Pheromone.MATERIAL = ETHANOL
###################################################################################

###################################################################################
# Action settings
###################################################################################
MySettings.Action.HIGH = 1.0
MySettings.Action.MEDIUM = 0.5
MySettings.Action.LOW = 0.2

###################################################################################
# Storage settings
###################################################################################
MySettings.Storage.ASSET_DIRECTORY = "./assets"
MySettings.Storage.SAVE_INDIVIDUALS = True
MySettings.Storage.SAVE_DIRECTORY = "results"
MySettings.Storage.SAVE_INTERVAL = 1
MySettings.Storage.TOP_N = 5
###################################################################################
