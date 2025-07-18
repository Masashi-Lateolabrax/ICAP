import numpy as np
from framework.prelude import Settings, calc_loss_sigma, Position, RobotLocation


def create_my_settings():
    settings = Settings()

    ###################################################################################
    # Server settings
    ###################################################################################
    settings.Server.HOST = "localhost"
    settings.Server.PORT = 5000
    settings.Server.SOCKET_BACKLOG = 10

    ###################################################################################
    # Render settings
    ###################################################################################
    settings.Render.RENDER_WIDTH = 500
    settings.Render.RENDER_HEIGHT = 500

    settings.Render.LIGHT_AMBIENT = 1.0
    settings.Render.LIGHT_DIFFUSE = 1.0
    settings.Render.LIGHT_SPECULAR = 1.0

    settings.Render.CAMERA_POS = (0.0, -1e-3, 13.0)
    settings.Render.CAMERA_LOOKAT = (0.0, 0.0, 0.0)

    ###################################################################################
    # Optimization settings
    ###################################################################################
    settings.Optimization.DIMENSION = None  # Will be set later based on the neural network dimension
    settings.Optimization.POPULATION = 100
    settings.Optimization.GENERATION = 500
    settings.Optimization.SIGMA = 0.1

    ###################################################################################
    # Robot settings
    ###################################################################################
    settings.Robot.HEIGHT = 0.1
    settings.Robot.RADIUS = 0.175
    settings.Robot.DISTANCE_BETWEEN_WHEELS = 0.175 * 2 * 0.8
    settings.Robot.MAX_SPEED = 0.8
    settings.Robot.MASS = 10

    settings.Robot.COLOR = (1, 1, 0, 1)

    settings.Robot.THINK_INTERVAL = 0.05

    settings.Robot.ACTUATOR_MOVE_KV = 100
    settings.Robot.ACTUATOR_ROT_KV = 10

    settings.Robot.ROBOT_SENSOR_GAIN = 1.0
    settings.Robot.FOOD_SENSOR_GAIN = 1.0

    settings.Robot.NUM = 9
    settings.Robot.INITIAL_POSITION = [
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
    # Food settings
    ###################################################################################
    settings.Food.RADIUS = 0.5
    settings.Food.HEIGHT = 0.07

    settings.Food.DENSITY = 80
    settings.Food.COLOR = (0, 1, 1, 1)

    settings.Food.NUM = 2
    settings.Food.INITIAL_POSITION = [
        Position(2, 2),
        Position(-2, 2),
    ]

    ###################################################################################
    # Nest settings
    ###################################################################################
    settings.Nest.POSITION = Position(0.0, 0.0)
    settings.Nest.RADIUS = 1.0
    settings.Nest.HEIGHT = 0.01
    settings.Nest.COLOR = (0, 1, 0, 1)

    ###################################################################################
    # Loss settings
    ###################################################################################
    settings.Loss.OFFSET_NEST_AND_FOOD = 0
    settings.Loss.SIGMA_NEST_AND_FOOD = calc_loss_sigma(4, 0.01)
    settings.Loss.GAIN_NEST_AND_FOOD = 1

    settings.Loss.OFFSET_ROBOT_AND_FOOD = settings.Robot.RADIUS + settings.Food.RADIUS
    settings.Loss.SIGMA_ROBOT_AND_FOOD = calc_loss_sigma(1, 0.3)
    settings.Loss.GAIN_ROBOT_AND_FOOD = 0.01

    settings.Loss.REGULARIZATION_COEFFICIENT = 0

    ###################################################################################
    # Simulation settings
    ###################################################################################
    settings.Simulation.TIME_STEP = 0.01
    settings.Simulation.TIME_LENGTH = 45

    settings.Simulation.WORLD_WIDTH = 10.0
    settings.Simulation.WORLD_HEIGHT = 10.0

    settings.Simulation.WALL_THICKNESS = 1
    settings.Simulation.WALL_HEIGHT = 1

    ###################################################################################
    # Storage settings
    ###################################################################################
    settings.Storage.SAVE_INDIVIDUALS = True
    settings.Storage.SAVE_DIRECTORY = "results"
    settings.Storage.SAVE_INTERVAL = 1
    settings.Storage.TOP_N = 5

    return settings


MySettings = create_my_settings
