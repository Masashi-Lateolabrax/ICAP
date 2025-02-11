class _Settings:
    RENDER_WIDTH = 500
    RENDER_HEIGHT = 500

    WORLD_WIDTH = 10
    WORLD_HEIGHT = 10

    ROBOT_SIZE = 0.175
    ROBOT_WEIGHT = 30  # kg
    ROBOT_MOVE_SPEED = 0.8
    ROBOT_TURN_SPEED = 3.14 / 2

    FOOD_SIZE = 0.5
    FOOD_FRICTIONLOSS = 1500

    SENSOR_GAIN = 1

    @property
    def SENSOR_OFFSET(self):
        return self.ROBOT_SIZE + self.FOOD_SIZE


Settings = _Settings()
