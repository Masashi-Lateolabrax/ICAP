import argparse
import os
import threading
import time
from datetime import datetime

from icecream import ic
import numpy as np
import torch

from framework.prelude import Settings, Individual
from framework.interfaces import SensorInterface
from framework.sensor import PreprocessedOmniSensor, DirectionSensor
from framework.optimization import connect_to_server

from src import utils
from settings import MySettings

ic.configureOutput(
    prefix=lambda: f'[{datetime.now().strftime("%H:%M:%S.%f")[:-3]}][PID:{os.getpid()}][TID:{threading.get_ident()}] CLIENT| ',
    includeContext=True
)

ic.disable()


class Loss(utils.Loss):
    def __init__(self, settings: Settings, robot_positions, food_positions, nest_position):
        self.settings = settings
        robot_pos_array = np.array(robot_positions)
        food_pos_array = np.array(food_positions)
        nest_pos_array = nest_position[0:2]
        self.r_loss = self._calc_r_loss(robot_pos_array, food_pos_array)
        self.n_loss = self._calc_n_loss(nest_pos_array, food_pos_array)

    def _calc_r_loss(self, robot_pos: np.ndarray, food_pos: np.ndarray) -> float:
        subs = (robot_pos[:, None, :] - food_pos[None, :, :]).reshape(-1, 2)
        distance = np.clip(
            np.linalg.norm(subs, axis=1) - self.settings.Loss.OFFSET_ROBOT_AND_FOOD,
            a_min=0,
            a_max=None
        )
        loss = -np.sum(np.exp(-(distance ** 2) / self.settings.Loss.SIGMA_ROBOT_AND_FOOD))
        return self.settings.Loss.GAIN_ROBOT_AND_FOOD * loss

    def _calc_n_loss(self, nest_pos: np.ndarray, food_pos: np.ndarray) -> float:
        subs = food_pos - nest_pos[None, :]
        distance = np.clip(
            np.linalg.norm(subs, axis=1) - self.settings.Loss.OFFSET_NEST_AND_FOOD,
            a_min=0,
            a_max=None
        )
        loss = -np.sum(np.exp(-(distance ** 2) / self.settings.Loss.SIGMA_NEST_AND_FOOD))
        return self.settings.Loss.GAIN_NEST_AND_FOOD * loss

    def as_float(self) -> float:
        return self.r_loss + self.n_loss


class RobotNeuralNetwork(torch.nn.Module):
    def __init__(self, parameters: Individual = None):
        super(RobotNeuralNetwork, self).__init__()

        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(6, 3),
            torch.nn.Tanhshrink(),
            torch.nn.Linear(3, 2),
            torch.nn.Tanh()
        )

        if parameters is not None:
            assert len(parameters) == self.dim, "Parameter length does not match the network's parameter count."
            torch.nn.utils.vector_to_parameters(
                torch.tensor(parameters, dtype=torch.float32),
                self.parameters()
            )

    @property
    def dim(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_):
        return self.sequential(input_)


class Simulator(utils.Simulator):
    def create_sensors(self) -> list[list[SensorInterface]]:
        sensors = []
        for i, robot in enumerate(self.robot_values):
            sensor_tuple = [
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
            ]
            sensors.append(sensor_tuple)
        return sensors

    def create_input_for_controller(self):
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


class Handler:
    def __init__(self):
        self.time = -1

    def run(self, individuals: list[Individual]):
        current_time = time.time()
        throughput = len(individuals) / ((current_time - self.time) + 1e-10)
        self.time = current_time

        ave_fitness = sum([i.get_fitness() for i in individuals]) / len(individuals)

        print(
            f"pid:{os.getpid()} "
            f"num: {len(individuals)} "
            f"fitness:{ave_fitness} "
            f"throughput:{throughput:.2f} ind/s"
        )


def main():
    parser = argparse.ArgumentParser(description="ICAP Optimization Client")
    parser.add_argument("--host", type=str, help="Server host address")
    parser.add_argument("--port", type=int, help="Server port number")
    args = parser.parse_args()

    settings = MySettings()

    host = args.host if args.host is not None else settings.Server.HOST
    port = args.port if args.port is not None else settings.Server.PORT

    print("=" * 50)
    print("OPTIMIZATION CLIENT")
    print("=" * 50)
    print(f"Server: {host}:{port}")
    print("-" * 30)
    print("Connecting to server...")
    print("Press Ctrl+C to disconnect")
    print("=" * 50)

    handler = Handler()

    connect_to_server(
        host,
        port,
        evaluation_function=EvaluationFunction(
            settings, lambda ind: Simulator(settings, ind)
        ).run,
        handler=handler.run,
    )


if __name__ == "__main__":
    main()
