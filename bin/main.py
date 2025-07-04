import mujoco
import numpy as np
import torch

from framework.optimization import connect_to_server
from framework.backends import MujocoBackend
from framework.utils import GenericTkinterViewer
from framework.sensor import PreprocessedOmniSensor, DirectionSensor
from framework.prelude import Settings, RobotLocation, Position, SensorInterface, RobotValues, Individual


class RobotNeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(RobotNeuralNetwork, self).__init__()
        self.linear1 = torch.nn.Linear(6, 3)
        self.activation1 = torch.nn.Tanhshrink()

        self.linear2 = torch.nn.Linear(3, 2)
        self.activation2 = torch.nn.Tanh()

    def forward(self, input_):
        x = self.linear1.forward(input_)
        x = self.activation1.forward(x)
        x = self.linear2.forward(x)
        x = self.activation2.forward(x)
        return x

    @property
    def dim(self):
        return sum(p.numel() for p in self.parameters())

    def set_parameters(self, parameters: Individual):
        assert len(parameters) == self.dim, \
            "Parameter length does not match the network's parameter count."

        torch.nn.utils.vector_to_parameters(
            torch.tensor(parameters, dtype=torch.float32),
            self.parameters()
        )


class SampleMujocoBackend(MujocoBackend):
    def __init__(self, settings: Settings, render: bool = False):
        super().__init__(settings, render)
        self.scores = []
        self.sensors: list[tuple[SensorInterface]] = self._create_sensors()

        self.controller = RobotNeuralNetwork()
        self.input_ndarray = np.zeros((settings.Robot.NUM, 3 * 2), dtype=np.float32)
        self.input_tensor = torch.from_numpy(self.input_ndarray)

        mujoco.mj_step(self.model, self.data)

    def _create_sensors(self) -> list[tuple[SensorInterface]]:
        sensors = []
        for i, robot in enumerate(self.robot_values):
            sensor_tuple = (
                PreprocessedOmniSensor(
                    robot,
                    self.settings.Robot.ROBOT_SENSOR_GAIN,
                    self.settings.Robot.RADIUS * 2,
                    [other.site for j, other in enumerate(self.robot_values) if j != i]
                ),
                PreprocessedOmniSensor(
                    robot,
                    self.settings.Robot.FOOD_SENSOR_GAIN,
                    self.settings.Robot.RADIUS + self.settings.Food.RADIUS,
                    [food.site for food in self.food_values]
                ),
                DirectionSensor(
                    robot, self.nest_site, self.settings.Nest.RADIUS
                )
            )
            sensors.append(sensor_tuple)
        return sensors

    def _create_input_for_controller(self):
        for i, sensors in enumerate(self.sensors):
            self.input_ndarray[i, 0:2] = sensors[0].get()
            self.input_ndarray[i, 2:4] = sensors[1].get()
            self.input_ndarray[i, 4:6] = sensors[2].get()
        return self.input_tensor

    def step(self):
        with torch.no_grad():
            input_ = self._create_input_for_controller()
            output = self.controller.forward(input_)
            output_ndarray = output.numpy()

        for i, robot in enumerate(self.robot_values):
            robot.act(
                right_wheel=output_ndarray[i, 0],
                left_wheel=output_ndarray[i, 1]
            )

        mujoco.mj_step(self.model, self.data)

    def score(self) -> list[float]:
        return self.scores

    Returns:
        float: objective function value
    """
    total = 0.0
    for i in range(individual.shape[0] - 1):
        total += 100.0 * (individual[i + 1] - individual[i] ** 2) ** 2 + (1 - individual[i]) ** 2
    return total


def main():
    settings = Settings()

    print("=" * 50)
    print("OPTIMIZATION CLIENT")
    print("=" * 50)
    print(f"Server: {settings.Server.HOST}:{settings.Server.PORT}")
    print("-" * 30)
    print("Connecting to server...")
    print("Press Ctrl+C to disconnect")
    print("=" * 50)

    connect_to_server(
        settings.Server.HOST,
        settings.Server.PORT,
        evaluation_function=rosenbrock_function,
    )


if __name__ == "__main__":
    main()
