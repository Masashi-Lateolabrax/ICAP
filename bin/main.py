import mujoco
import numpy as np
import torch

from framework.optimization import connect_to_server
from framework.backends import MujocoBackend
from framework.sensor import PreprocessedOmniSensor, DirectionSensor
from framework.prelude import Settings, RobotLocation, Position, SensorInterface, RobotValues, Individual


class RobotNeuralNetwork(torch.nn.Module):
    def __init__(self, parameters: Individual):
        assert len(parameters) == self.dim, "Parameter length does not match the network's parameter count."

        super(RobotNeuralNetwork, self).__init__()

        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(6, 3),
            torch.nn.Tanhshrink(),
            torch.nn.Linear(3, 2),
            torch.nn.Tanh()
        )

        torch.nn.utils.vector_to_parameters(
            torch.tensor(parameters, dtype=torch.float32),
            self.parameters()
        )

    @property
    def dim(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_):
        return self.sequential(input_)


class Loss:
    def __init__(self, settings: Settings, robot_positions, food_positions, nest_position):
        self.settings = settings
        self.r_loss = self._calc_r_loss(robot_positions, food_positions)
        self.n_loss = self._calc_n_loss(nest_position, food_positions)

    def _calc_r_loss(self, robot_pos: np.ndarray, food_pos: np.ndarray) -> float:
        subs = (robot_pos[:, None, :2] - food_pos[None, :, :]).reshape(-1, 2)
        distance = np.clip(
            np.linalg.norm(subs, axis=1) - self.settings.Loss.OFFSET_ROBOT_AND_FOOD,
            a_min=0,
            a_max=None
        )
        loss = np.sum(np.exp(-(distance ** 2) / self.settings.Loss.SIGMA_ROBOT_AND_FOOD))
        return self.settings.Loss.GAIN_ROBOT_AND_FOOD * loss

    def _calc_n_loss(self, nest_pos: np.ndarray, food_pos: np.ndarray) -> float:
        subs = food_pos - nest_pos
        distance = np.clip(
            np.linalg.norm(subs, axis=1) - self.settings.Loss.OFFSET_NEST_AND_FOOD,
            a_min=0,
            a_max=None
        )
        loss = np.sum(np.exp(-(distance ** 2) / self.settings.Loss.OFFSET_NEST_AND_FOOD))
        return self.settings.Loss.GAIN_NEST_AND_FOOD * loss

    def as_float(self) -> float:
        return self.r_loss + self.n_loss


class Simulation(MujocoBackend):
    def __init__(self, settings: Settings, parameters: Individual, render: bool = False):
        super().__init__(settings, render)

        self.parameters = parameters
        self.scores: list[Loss] = []
        self.sensors: list[tuple[SensorInterface]] = self._create_sensors()

        self.controller = RobotNeuralNetwork(parameters)
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

    def evaluation(self) -> Loss:
        robot_positions = [r.xpos for r in self.robot_values]
        food_positions = [f.xpos for f in self.food_values]
        nest_position = self.nest_site.xpos
        return Loss(
            self.settings,
            robot_positions=robot_positions,
            food_positions=food_positions,
            nest_position=nest_position
        )

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

        loss = self.evaluation()
        self.scores.append(loss)

    def scores(self) -> list[float]:
        return [s.as_float() for s in self.scores]

    def total_score(self) -> float:
        regularization_loss = self.settings.Optimization.REGULARIZATION_LOSS * self.parameters.norm
        return sum(s.as_float() for s in self.scores) + regularization_loss

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
